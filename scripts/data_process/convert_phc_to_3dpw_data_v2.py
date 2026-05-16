import argparse
import json
import os
import os.path as osp
import pickle
import sys
import warnings

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


R_3DPW_TO_PHC = sRot.from_euler("x", 90, degrees=True)
R_PHC_TO_3DPW = R_3DPW_TO_PHC.inv()
Q_UPRIGHT_ALIGN = sRot.from_quat([0.5, 0.5, 0.5, 0.5])


def to_numpy(x):
    """Accept numpy arrays / torch tensors from PHC joblib files."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def smpl_mujoco_maps():
    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(name)
        for name in SMPL_MUJOCO_NAMES
        if name in SMPL_BONE_ORDER_NAMES
    ]
    mujoco_2_smpl = np.argsort(smpl_2_mujoco)
    return smpl_2_mujoco, mujoco_2_smpl


def gender_to_number(gender):
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ["male", "m"]:
        return [1]
    if gender in ["female", "f"]:
        return [2]
    return [0]


def build_smpl_skeleton_tree(beta=None, gender="neutral", tmp_xml_path=None):
    """Build the same SMPL MuJoCo skeleton topology used by convert_3dpw_data_v2.py."""
    if beta is None:
        beta = np.zeros(16, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    if beta.shape[0] < 16:
        beta_16 = np.zeros(16, dtype=np.float32)
        beta_16[: beta.shape[0]] = beta
        beta = beta_16

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "replace_feet": True,
        "big_ankle": True,
        "model": "smpl",
        "body_params": {},
    }
    if tmp_xml_path is None:
        tmp_xml_path = "phc/data/assets/mjcf/smpl_tmp_inverse_3dpw.xml"
    os.makedirs(osp.dirname(tmp_xml_path), exist_ok=True)

    smpl_local_robot = LocalRobot(robot_cfg)
    smpl_local_robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]), gender=gender_to_number(gender), objs_info=None
    )
    smpl_local_robot.write_xml(tmp_xml_path)
    return SkeletonTree.from_mjcf(tmp_xml_path)


def phc_points_to_3dpw(points_phc, meta, upright_start=True, to_camera=True):
    """
    Reverse the point-coordinate part of convert_3dpw_data_v2.py.

    Args:
        points_phc: (N, J, 3), PHC/MuJoCo joint positions, usually pos_state/pred_pos.
        meta: one motion entry from data/3dpw/3dpw_*_gender_ground.pkl.
        upright_start: whether the forward script used --upright_start.
        to_camera: if True, return original 3DPW camera coordinates; otherwise 3DPW world coords.

    Returns:
        (N, J, 3) in 3DPW SMPL joint order.
    """
    points_phc = to_numpy(points_phc).astype(np.float64)
    n = points_phc.shape[0]

    root_trans_offset = to_numpy(meta["root_trans_offset"]).astype(np.float64)
    trans_orig_phc = to_numpy(meta["trans_orig"]).astype(np.float64)

    # 1) Undo PHC root offset / ground offset.  If upright_start was used, also undo
    #    the fixed upright alignment around the root.
    if upright_start:
        centered = points_phc - root_trans_offset[:, None, :]
        points_phc_unaligned = Q_UPRIGHT_ALIGN.apply(centered.reshape(-1, 3)).reshape(n, -1, 3)
        points_phc_unaligned = points_phc_unaligned + trans_orig_phc[:, None, :]
    else:
        points_phc_unaligned = points_phc + (trans_orig_phc - root_trans_offset)[:, None, :]

    # 2) PHC/AMASS coordinate -> 3DPW world coordinate.
    points_3dpw = R_PHC_TO_3DPW.apply(points_phc_unaligned.reshape(-1, 3)).reshape(n, -1, 3)

    # 3) MuJoCo joint order -> SMPL joint order.
    _, mujoco_2_smpl = smpl_mujoco_maps()
    points_3dpw = points_3dpw[:, mujoco_2_smpl, :]

    # 4) 3DPW world coordinate -> original 3DPW camera coordinate.
    if to_camera:
        cam_poses = to_numpy(meta["cam_poses"]).astype(np.float64)
        r_cam = cam_poses[:n, :3, :3]
        t_cam = cam_poses[:n, :3, 3]
        # Forward in convert_3dpw_data_v2.py:
        #     world = R_cam @ camera + t_cam
        # Therefore the inverse is:
        #     camera = R_cam.T @ (world - t_cam)
        points_3dpw = np.einsum(
            "nij,nkj->nki",
            np.transpose(r_cam, (0, 2, 1)),
            points_3dpw - t_cam[:, None, :],
        )

    return points_3dpw


def phc_root_trans_to_3dpw(meta, to_camera=True):
    """Recover SMPL root translation saved by 3DPW, i.e. inverse of root_trans conversion."""
    trans_orig_phc = to_numpy(meta["trans_orig"]).astype(np.float64)
    n = trans_orig_phc.shape[0]
    trans_3dpw = R_PHC_TO_3DPW.apply(trans_orig_phc)

    if to_camera:
        cam_poses = to_numpy(meta["cam_poses"]).astype(np.float64)
        r_cam = cam_poses[:n, :3, :3]
        t_cam = cam_poses[:n, :3, 3]
        trans_3dpw = np.einsum(
            "nij,nj->ni",
            np.transpose(r_cam, (0, 2, 1)),
            trans_3dpw - t_cam,
        )

    return trans_3dpw


def phc_pose_quat_to_3dpw_pose_aa(meta, upright_start=True, skeleton_tree=None):
    """
    Reverse local quaternion pose conversion and output 3DPW-like SMPL axis-angle pose.

    Note: convert_3dpw_data_v2.py drops the last two SMPL hand joints by replacing them
    with zeros, so this inverse also outputs zeros for those 6 axis-angle values.
    """
    pose_quat = to_numpy(meta["pose_quat"]).astype(np.float64)
    n = pose_quat.shape[0]

    if upright_start:
        if skeleton_tree is None:
            skeleton_tree = build_smpl_skeleton_tree(meta.get("beta"), meta.get("gender", "neutral"))
        zero_root = torch.zeros((n, 3), dtype=torch.float64)
        aligned_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat).to(torch.float64),
            zero_root,
            is_local=True,
        )
        global_aligned = to_numpy(aligned_state.global_rotation)
        global_orig = (
            sRot.from_quat(global_aligned.reshape(-1, 4)) * Q_UPRIGHT_ALIGN
        ).as_quat().reshape(n, -1, 4)
        orig_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(global_orig).to(torch.float64),
            zero_root,
            is_local=False,
        )
        pose_quat = to_numpy(orig_state.local_rotation)

    _, mujoco_2_smpl = smpl_mujoco_maps()
    pose_quat_smpl = pose_quat[:, mujoco_2_smpl, :]
    pose_aa = sRot.from_quat(pose_quat_smpl.reshape(-1, 4)).as_rotvec().reshape(n, 24, 3)

    # Root orientation: PHC/AMASS -> 3DPW.
    root_phc = sRot.from_rotvec(pose_aa[:, 0, :])
    pose_aa[:, 0, :] = (R_PHC_TO_3DPW * root_phc).as_rotvec()
    return pose_aa.reshape(n, 72)


def attach_action_positions(meta_dict, action_path):
    """Merge PHC rollout positions into the metadata dict produced by convert_3dpw_data_v2.py."""
    action_data = joblib.load(action_path)
    keys = action_data["key_names"]
    pred_pos = action_data.get("pred_pos")
    gt_pos = action_data.get("gt_pred_pos")

    for idx, key in enumerate(keys):
        if key not in meta_dict:
            warnings.warn(f"{key} not found in PHC metadata; skipped.", RuntimeWarning)
            continue
        if pred_pos is not None:
            meta_dict[key]["pos_state"] = pred_pos[idx]
        if gt_pos is not None:
            meta_dict[key]["gt_pos_state"] = gt_pos[idx]
    return meta_dict


def inverse_conversion(val, key, args):
    """
    与 convert_inverse_3dpw_from_PHC.py 保持一致的单条 motion 逆转换入口。

    Args:
        val: convert_3dpw_data_v2.py 保存的一条 PHC meta，并已补入 pos_state/gt_pos_state。
        key: "pos_state" 或 "gt_pos_state"。
        args: 命令行参数。

    Returns:
        (N, 24, 3)，3DPW SMPL joint order，默认回到原始 3DPW camera 坐标系。
    """
    return phc_points_to_3dpw(
        val[key],
        val,
        upright_start=args.upright_start,
        to_camera=args.to_camera,
    )


def convert_file(args):
    """
    输入/输出格式对齐 scripts/data_process/convert_inverse_3dpw_from_PHC.py：

    Input:
        --phc_stat_name: convert_3dpw_data_v2.py 生成的 PHC metadata pkl
        --phc_act_stat_name: PHC rollout pkl，包含 pred_pos/gt_pred_pos/key_names

    Output:
        {
            "rl_pos": {motion_key: restored_pred_joints},
            "rl_gt_pos": {motion_key: restored_gt_joints},
        }
    """
    phc_act_stat_data = joblib.load(args.phc_act_stat_name)
    phc_act_pos = phc_act_stat_data["pred_pos"]
    phc_gt_act_pos = phc_act_stat_data["gt_pred_pos"]
    phc_act_key = phc_act_stat_data["key_names"]

    phc_act_stat = dict(zip(phc_act_key, phc_act_pos))
    phc_gt_act_stat = dict(zip(phc_act_key, phc_gt_act_pos))

    phc_stat = joblib.load(args.phc_stat_name)
    phc_stat_full = phc_stat.copy()
    for key, value in phc_act_stat.items():
        if key in phc_stat_full:
            phc_stat_full[key].update({"pos_state": value})
            phc_stat_full[key].update({"gt_pos_state": phc_gt_act_stat[key]})
        else:
            warnings.warn(f"{key} not found in PHC metadata; skipped.", RuntimeWarning)

    restored_motions, restored_gt_motions = {}, {}
    for key, val in tqdm(phc_stat_full.items(), desc="PHC -> 3DPW"):
        if "pos_state" in val:
            restored_motions[key] = inverse_conversion(val, key="pos_state", args=args)
        if "gt_pos_state" in val:
            restored_gt_motions[key] = inverse_conversion(val, key="gt_pos_state", args=args)

    data = {
        "rl_pos": restored_motions,
        "rl_gt_pos": restored_gt_motions,
    }

    output_path = args.output
    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(data, output_path)
    # print("????????")
    print(f"Saved to {output_path}")

    if args.validate:
        report = validate_against_vanilla(
            data,
            vanilla_path=args.vanilla_path,
            split=args.process_split,
            atol=args.atol,
            rtol=args.rtol,
        )
        print_validation_report(report)
        if args.validation_output:
            validation_dir = osp.dirname(args.validation_output)
            if validation_dir:
                os.makedirs(validation_dir, exist_ok=True)
            with open(args.validation_output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Validation report saved to {args.validation_output}")


def motion_key_to_seq_person(key):
    """Split a key such as 'downtown_arguing_00_1' into sequence name and person id."""
    seq_name, person_id = key.rsplit("_", 1)
    return seq_name, int(person_id)


def get_vanilla_joints(vanilla_cache, vanilla_path, split, key):
    """Return original 3DPW joints as (N, 24, 3) in original camera space."""
    seq_name, person_id = motion_key_to_seq_person(key)
    if seq_name not in vanilla_cache:
        pkl_path = osp.join(vanilla_path, "sequence", split, f"{seq_name}.pkl")
        with open(pkl_path, "rb") as f:
            vanilla_cache[seq_name] = pickle.load(f, encoding="latin1")

    seq_data = vanilla_cache[seq_name]
    joint_field = "jointPositions" if "jointPositions" in seq_data else "joint_positions"
    joints = np.asarray(seq_data[joint_field][person_id], dtype=np.float64)
    return joints.reshape(joints.shape[0], 24, 3)


def compare_motion(restored, target, atol, rtol):
    n = min(restored.shape[0], target.shape[0])
    diff = np.asarray(restored[:n], dtype=np.float64) - np.asarray(target[:n], dtype=np.float64)
    joint_l2 = np.linalg.norm(diff, axis=-1)
    return {
        "num_frames_compared": int(n),
        "restored_frames": int(restored.shape[0]),
        "vanilla_frames": int(target.shape[0]),
        "allclose": bool(np.allclose(restored[:n], target[:n], atol=atol, rtol=rtol)),
        "max_abs_coord_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_coord_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse_coord": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
        "max_joint_l2_error": float(np.max(joint_l2)) if joint_l2.size else 0.0,
        "mean_joint_l2_error": float(np.mean(joint_l2)) if joint_l2.size else 0.0,
    }


def summarize_motion_reports(motion_reports):
    if not motion_reports:
        return {"num_motions": 0, "num_allclose": 0, "all_motions_allclose": False}

    return {
        "num_motions": len(motion_reports),
        "num_allclose": int(sum(r["allclose"] for r in motion_reports.values())),
        "all_motions_allclose": bool(all(r["allclose"] for r in motion_reports.values())),
        "max_abs_coord_error": float(max(r["max_abs_coord_error"] for r in motion_reports.values())),
        "mean_abs_coord_error": float(np.mean([r["mean_abs_coord_error"] for r in motion_reports.values()])),
        "rmse_coord": float(np.sqrt(np.mean([r["rmse_coord"] ** 2 for r in motion_reports.values()]))),
        "max_joint_l2_error": float(max(r["max_joint_l2_error"] for r in motion_reports.values())),
        "mean_joint_l2_error": float(np.mean([r["mean_joint_l2_error"] for r in motion_reports.values()])),
    }


def validate_against_vanilla(restored_data, vanilla_path, split="test", atol=1e-5, rtol=1e-5):
    """Compare restored PHC rollout coordinates with original 3DPW jointPositions."""
    vanilla_cache = {}
    report = {
        "vanilla_path": vanilla_path,
        "split": split,
        "atol": atol,
        "rtol": rtol,
        "fields": {},
    }

    for field_name in ["rl_gt_pos", "rl_pos"]:
        motion_reports = {}
        for key, restored in restored_data.get(field_name, {}).items():
            target = get_vanilla_joints(vanilla_cache, vanilla_path, split, key)
            motion_reports[key] = compare_motion(np.asarray(restored), target, atol, rtol)

        report["fields"][field_name] = {
            "summary": summarize_motion_reports(motion_reports),
            "motions": motion_reports,
        }

    return report


def print_validation_report(report):
    print("\nValidation against original 3DPW jointPositions:")
    for field_name, field_report in report["fields"].items():
        summary = field_report["summary"]
        print(
            f"  {field_name}: allclose {summary['num_allclose']}/{summary['num_motions']}, "
            f"max_joint_l2={summary.get('max_joint_l2_error', 0.0):.6g}, "
            f"mean_joint_l2={summary.get('mean_joint_l2_error', 0.0):.6g}, "
            f"rmse_coord={summary.get('rmse_coord', 0.0):.6g}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inverse of scripts/data_process/convert_3dpw_data_v2.py: PHC coords -> 3DPW coords."
    )
    parser.add_argument(
        "--phc_stat_name",
        type=str,
        default="data/3dpw/3dpw_test_upright_whole_sequence_gender_ground.pkl",
        help="Metadata pkl produced by convert_3dpw_data_v2.py.",
    )
    parser.add_argument(
        "--phc_act_stat_name",
        type=str,
        default="sample_data/phc_act/3dpw_test_upright_whole_sequence_gender_ground/noise_False_0.05.pkl",
        help="PHC rollout pkl containing pred_pos/gt_pred_pos/key_names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=osp.join(os.getcwd(), "data", "3dpw", "data_offline_multi3dpw.pkl"),
    )
    parser.add_argument(
        "--vanilla_path",
        type=str,
        default="data/3dpw/Vanilla",
        help="Original 3DPW Vanilla root used for validation.",
    )
    parser.add_argument("--process_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate restored coordinates against Vanilla jointPositions.",
    )
    parser.add_argument(
        "--validation_output",
        type=str,
        default=osp.join(os.getcwd(), "data", "3dpw", "phc_to_3dpw_validation.json"),
        help="Where to save the JSON validation report when --validate is set.",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for allclose validation.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for allclose validation.")
    parser.add_argument("--gender_flag", action="store_true", default=True)
    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument(
        "--no_upright_start",
        dest="upright_start",
        action="store_false",
        help="Use this only if the forward conversion did not use --upright_start.",
    )
    parser.add_argument("--on_the_ground", action="store_true", default=True)
    parser.add_argument(
        "--to_world",
        dest="to_camera",
        action="store_false",
        help="Return 3DPW world coordinates instead of original 3DPW camera coordinates.",
    )
    # parser.add_argument(
    #     "--to_camera",
    #     default=True,
    #     action="store_false",
    #     help="Return 3DPW world coordinates instead of original 3DPW camera coordinates.",
    # )
    args = parser.parse_args()
    convert_file(args)
