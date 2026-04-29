import glob
import os
import sys
import os.path as osp
import torch
import numpy as np
import joblib
import pickle
import warnings
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation as sRot
from collections import defaultdict

sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


def inverse_conversion(val, args):
    """
    逆向转换函数
    :param val: phc_stat_full 中的单个字典元素
    :param args: 参数
    :return: 恢复到原始 3DPW 相机坐标系下的 3D 关节坐标 (N, 24, 3)
    """
    joints_inv = val["pos_state"].copy() # (N, 24, 3)
    N = joints_inv.shape[0]
    
    # 提取保存好的平移状态
    root_trans_offset = val['root_trans_offset'].numpy() # (N, 3)，包含了 local_translation 和 floor_offset
    trans_orig = val['trans_orig']               # (N, 3)，纯净的 AMASS 坐标系下的根节点平移

    # 1. 消除所有平移偏移 & 还原 upright_start 旋转
    if args.upright_start:
        q_align = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
        
        # 必须以带有所有偏移的 root_trans_offset 为中心点进行逆向旋转
        joints_centered_root = joints_inv - root_trans_offset[:, np.newaxis, :]
        reshaped_joints = joints_centered_root.reshape(-1, 3)
        
        # 执行逆向旋转 (此时是绕着角色自己的 root 旋转)
        joints_inv = q_align.apply(reshaped_joints).reshape(N, 24, 3)
        
        # 旋转完成后，加上最初始、纯净的 AMASS 坐标平移，直接绕过了骨骼偏移和地面偏移的影响
        joints_inv = joints_inv + trans_orig[:, np.newaxis, :]
    else:
        # 如果没有执行 upright_start，只需把偏移的差值补回去即可
        offset_diff = trans_orig - root_trans_offset
        joints_inv = joints_inv + offset_diff[:, np.newaxis, :]

    # 2. AMASS 坐标系还原回 3DPW 坐标系 (绕 X 轴旋转 -90 度)
    R_amass_2_3dpw = sRot.from_euler('x', -90, degrees=True)
    reshaped_joints = joints_inv.reshape(-1, 3)
    joints_inv = R_amass_2_3dpw.apply(reshaped_joints).reshape(N, 24, 3)
    
    # 3. 还原关节点顺序 (Mujoco 顺序 -> SMPL 顺序)
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    mujoco_2_smpl = np.argsort(smpl_2_mujoco)
    joints_inv = joints_inv[:, mujoco_2_smpl, :] # 确保保持 3 维特征
    
    # 4. 世界坐标系还原回原始相机坐标系
    cam_poses = val['cam_poses']
    R_cam, t_cam = cam_poses[:, :3, :3], cam_poses[:, :3, 3]  
    
    # 先减去平移 t_cam (注意增加维度用于 broadcasting)
    joints_centered_cam = joints_inv - t_cam[:, np.newaxis, :]
    
    # 乘上相机旋转的转置 (R_cam^T) 实现逆向旋转
    # einsum 的 'nij,nkj->nki' 逻辑完全正确，等价于对每个 joint 做 R^T @ p
    R_cam_T = np.transpose(R_cam, (0, 2, 1))
    joints_original_cam = np.einsum('nij,nkj->nki', R_cam_T, joints_centered_cam)
    
    return joints_original_cam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phc_stat_name", type=str, default="data/3dpw/3dpw_test_upright_whole_sequence_gender_ground.pkl")
    # parser.add_argument("--phc_act_stat_name", type=str, default="sample_data/phc_act/3dpw_test_upright_whole_sequence_gender_ground/noise_False_0.05.pkl")
    parser.add_argument("--phc_act_stat_name", type=str, default="sample_data/phc_act/phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl")
    parser.add_argument("--gender_flag", action="store_true", default=True)
    parser.add_argument("--upright_start", action="store_true", default=True)
    parser.add_argument("--on_the_ground", action="store_true", default=True)
    args = parser.parse_args()
    
    phc_act_stat_data = joblib.load(args.phc_act_stat_name)
    phc_act_pos, phc_act_key = phc_act_stat_data['pred_pos'], phc_act_stat_data['key_names']
    phc_act_stat = dict(zip(phc_act_key, phc_act_pos))
    
    phc_stat = joblib.load(args.phc_stat_name)     
    phc_stat_full = phc_stat.copy()
    for key, value in phc_act_stat.items():
        if key in phc_stat_full:
            phc_stat_full[key].update({"pos_state": value})
        else:
            warnings.warn("当前值{}不存在!".format(key), RuntimeWarning)
            
    # 执行逆运算并保存/处理结果
    restored_motions = {}
    for key, val in phc_stat_full.items():        
        if "pos_state" in val:
            restored_motions[key] = inverse_conversion(val, args)   # 传入整个 val 字典，以便在函数内访问所有原始 offset 和相机数据
            
    print("坐标逆转换完成！")