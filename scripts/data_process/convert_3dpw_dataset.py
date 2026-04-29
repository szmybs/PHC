import glob
import os
import sys
import os.path as osp
import torch
import numpy as np
import joblib
import pickle
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation as sRot

# 假设你的项目结构
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

def process_3dpw(args):
    process_split = args.process_split
    upright_start = args.upright_start
    
    # 1. 定义 3DPW 到 AMASS 的坐标转换矩阵 (绕X轴旋转90度)
    R_3dpw_2_amass = sRot.from_euler('x', 90, degrees=True)
    
    robot_cfg = {
        "mesh": False, "rel_joint_lm": True, "upright_start": upright_start,
        "remove_toe": False, "real_weight": True, "replace_feet": True,
        "big_ankle": True, "model": "smpl", "body_params": {},
    }

    smpl_local_robot = LocalRobot(robot_cfg)
    data_path = osp.join(args.path, "sequence", process_split)
    
    all_pkls = glob.glob(f"{data_path}/*.pkl")
    pw3d_full_motion_dict = {}
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]

    for pkl_path in tqdm(all_pkls):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        seq_name = osp.basename(pkl_path).replace(".pkl", "")
        num_people = len(data['poses'])
        
        cam_poses = data['cam_poses']
        # 提取相机的旋转和位移
        R_cam = cam_poses[:, :3, :3]  # 所有帧的相机旋转 (N, 3, 3)
        t_cam = cam_poses[:, :3, 3]   # 所有帧的相机平移 (N, 3)
    
        for person_id in range(num_people):
            key_name = f"{seq_name}_{person_id}"
            
            # --- 数据提取 ---
            pose_aa_full = data['poses'][person_id] # (N, 72)
            root_trans = data['trans'][person_id]  # (N, 3)
            betas = data['betas'][person_id][:10]
            gender = data['genders'][person_id]
            
            # 2. 强制性别为 Neutral
            gender_str = "neutral"
            gender_num = [2] 
            
            N = pose_aa_full.shape[0]
            if N < 10: continue
            
            # --- 3. 相机坐标变换到世界坐标系 
            root_trans = np.einsum('nij,nj->ni', R_cam, root_trans) + t_cam
            # --- 3. 坐标系转换 (3DPW -> AMASS) ---
            # 转换平移
            root_trans_amass = R_3dpw_2_amass.apply(root_trans)
            
            # 转换根旋转 (Global Orientation)
            root_rot_obj = sRot.from_rotvec(pose_aa_full[:, :3])
            root_rot_amass = R_3dpw_2_amass * root_rot_obj

            pose_aa_amass = pose_aa_full.copy()
            pose_aa_amass[:, :3] = root_rot_amass.as_rotvec()
            
            # --- 4. 骨架生成 (基于 Neutral 模板) ---
            # 补齐手部关节并转换顺序
            pose_aa = np.concatenate([pose_aa_amass[:, :66], np.zeros((N, 6))], axis=-1)
            pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

            # 使用 Neutral 性别加载机器人，这会影响骨骼长度计算
            beta = np.zeros((16))
            gender_number, beta[:], gender = [0], 0, "neutral"
            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            
            # 写入临时 MJCF
            tmp_xml_path = f"phc/data/assets/mjcf/smpl_tmp_{person_id}.xml"
            os.makedirs(osp.dirname(tmp_xml_path), exist_ok=True)
            smpl_local_robot.write_xml(tmp_xml_path)
            skeleton_tree = SkeletonTree.from_mjcf(tmp_xml_path)
            
            # 结合平移
            root_trans_offset = torch.from_numpy(root_trans_amass) + skeleton_tree.local_translation[0]

            # 创建 SkeletonState
            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(pose_quat), root_trans_offset, is_local=True
            )
            
            # --- 5. 朝向对齐 (Upright Start) ---
            if upright_start:
                # 抵消起始旋转，应用标准对齐矩阵 [0.5, 0.5, 0.5, 0.5]
                # 这是为了让角色在 Z 向上的世界中面朝前方
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False
                )

            pose_quat_global_final = new_sk_state.global_rotation.numpy() 
            pose_quat_final = new_sk_state.local_rotation.numpy()
            trans_orig_final = root_trans_amass
            root_trans_offset_final = root_trans_offset
            pose_aa_final = pose_aa
            
            step = 30
            for i in range(0, N-step, 1):
                # 封装结果                
                new_key_name = f"{key_name}_{i}"
                pw3d_full_motion_dict[new_key_name] = {
                    'pose_quat_global': pose_quat_global_final[i: i+step],
                    'pose_quat': pose_quat_final[i: i+step],
                    'trans_orig': trans_orig_final[i: i+step],
                    'root_trans_offset': root_trans_offset_final[i: i+step],
                    'beta': beta,
                    'gender': gender_str, # 存储为 neutral
                    'pose_aa': pose_aa_final[i: i+step],
                    'fps': 30
                }

    # 保存
    output_dir = "data/3dpw"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_upright" if upright_start else ""
    output_file = osp.join(output_dir, f"3dpw_{process_split}{suffix}.pkl")
    joblib.dump(pw3d_full_motion_dict, output_file, compress=True)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/ThreeDPW")
    parser.add_argument("--process_split", type=str, default="test", choices=['train', 'val', 'test'])
    parser.add_argument("--upright_start", action="store_true", default=True)
    args = parser.parse_args()
    process_3dpw(args)