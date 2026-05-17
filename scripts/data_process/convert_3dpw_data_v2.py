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
    
    # 定义 3DPW 到 AMASS 的坐标转换矩阵 (绕X轴旋转90度)
    R_3dpw_2_amass = sRot.from_euler('x', 90, degrees=True)
    
    robot_cfg = {
        "mesh": False, 
        "rel_joint_lm": True, 
        "upright_start": upright_start, 
        "remove_toe": False, 
        "real_weight": True, 
        "replace_feet": True, 
        "big_ankle": True, 
        "model": "smpl", 
        "body_params": {},
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
        if num_people <= 1:
            continue
        
        cam_poses = data['cam_poses']
        # 提取相机的旋转和位移
        R_cam = cam_poses[:, :3, :3]  # 所有帧的相机旋转 (N, 3, 3)
        t_cam = cam_poses[:, :3, 3]   # 所有帧的相机平移 (N, 3)
    
        for person_id in range(num_people):
            key_name = f"{seq_name}_{person_id}"
            
            # 数据提取
            pose_aa_full = data['poses'][person_id] # (N, 72)
            root_trans = data['trans'][person_id]  # (N, 3)
            betas = data['betas'][person_id][:10]
            gender = data['genders'][person_id]
            
            if args.gender_flag:
                if gender == "neutral":
                    gender_str = "neutral"
                    gender_number = [0]
                elif gender == "male" or gender =='m':
                    gender_str = "male"
                    gender_number = [1]
                elif gender == "female" or gender == 'f':
                    gender_str = "female"
                    gender_number = [2]
                beta = np.zeros((16))
                beta[:10] = betas
            else: 
                # 强制性别为 Neutral
                gender_str = "neutral"
                gender_number = [2] 
                beta = np.zeros((16))
            
            N = pose_aa_full.shape[0]
            if N < 10: continue
            
            # 相机坐标变换到世界坐标系(hip节点坐标)     ### 这里或者是由世界坐标系转到相机坐标系？？？
            # root_trans = np.einsum('nij,nj->ni', R_cam, root_trans) + t_cam
            root_trans_amass = R_3dpw_2_amass.apply(root_trans)     # 坐标系转换 (3DPW -> AMASS)
            
            # 转换根旋转 (Global Orientation)
            # 3DPW 的 root orientation 和 root translation 一样先从相机坐标系
            # 变换到世界坐标系，再从 3DPW 世界坐标系变换到 AMASS/PHC 坐标系：
            #   R_root_world = R_cam @ R_root_cam
            #   R_root_amass = R_3dpw_2_amass @ R_root_world
            root_rot_cam = sRot.from_rotvec(pose_aa_full[:, :3])
            # root_rot_world = sRot.from_matrix(R_cam[:N]) * root_rot_cam   ### 到PHC中不对，倒立了
            # root_rot_amass = R_3dpw_2_amass * root_rot_world
            root_rot_amass = R_3dpw_2_amass * root_rot_cam

            
            pose_aa_amass = pose_aa_full.copy()
            pose_aa_amass[:, :3] = root_rot_amass.as_rotvec()
            
            # 骨架生成, 补齐手部关节并转换顺序
            pose_aa = np.concatenate([pose_aa_amass[:, :66], np.zeros((N, 6))], axis=-1)
            pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

            # 根据性别加载机器人
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
            
            # 朝向对齐 (Upright Start)
            if upright_start:
                # 抵消起始旋转，应用标准对齐矩阵 [0.5, 0.5, 0.5, 0.5]
                # 这是为了让角色在 Z 向上的世界中面朝前方
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False
                )
            
            if args.on_the_ground:
                up_axis_idx = 2 if args.upright_start else 1

                # 获取当前所有关节的世界坐标 (N, 24, 3)
                global_joints = new_sk_state.global_translation # 这是一个 Torch Tensor
                # 找到每一帧中所有关节的最低高度值 (N,)
                # min_heights 存储了每一帧人体“最深处”的坐标值
                min_heights, _ = torch.min(global_joints[..., up_axis_idx], dim=1)

                # 执行动态修正：
                # root_trans_offset 是 (N, 3)，我们将每一帧对应的向上轴位移减去该帧的最低高度
                root_trans_offset_fixed = root_trans_offset.clone()
                root_trans_offset_fixed[:, up_axis_idx] -= min_heights
                
                # 重新创建对齐地面的 SkeletonState
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, 
                    new_sk_state.local_rotation, # 保持局部旋转不变
                    root_trans_offset_fixed, 
                    is_local=True
                )
                root_trans_offset = root_trans_offset_fixed
            else:
                min_heights = torch.zeros(size=(1,))

            # 封装结果
            pw3d_full_motion_dict[key_name] = {
                'pose_quat_global': new_sk_state.global_rotation.numpy(),
                'pose_quat': new_sk_state.local_rotation.numpy(),
                'trans_orig': root_trans_amass,
                # 'root_trans_offset': root_trans_offset.numpy(),
                'root_trans_offset': root_trans_offset,
                'beta': beta,
                'gender': gender_str,
                'pose_aa': pose_aa,
                'fps': 30,
                'floor_offset': min_heights.numpy(),
                'cam_poses': data['cam_poses'],
            }

    # 保存
    output_dir = "data/3dpw"
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = "_upright_whole_sequence" if upright_start else ""
    suffix = suffix if not args.gender_flag else f"{suffix}_gender"
    suffix = suffix if not args.on_the_ground else f"{suffix}_ground"
    
    output_file = osp.join(output_dir, f"3dpw_{process_split}{suffix}.pkl")
    joblib.dump(pw3d_full_motion_dict, output_file, compress=True)
    print(f"Saved to {output_file}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/3dpw/Vanilla")
    parser.add_argument("--process_split", type=str, default="test", choices=['train', 'val', 'test'])
    parser.add_argument("--upright_start", action="store_true", default=True)
    parser.add_argument("--gender_flag", action="store_true", default=True)
    parser.add_argument("--on_the_ground", action="store_true", default=True)
    args = parser.parse_args()
    
    process_3dpw(args)
