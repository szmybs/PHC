"""
Validate 3DPW pose coordinate conversion + FK against PHC action positions.

功能：
1. 复现 scripts/data_process/convert_3dpw_data_v2.py 中的关键坐标变换：
   - root translation: 3DPW -> AMASS/PHC, Rx(+90deg)
   - root orientation: R_3DPW_TO_AMASS * R_root
   - SMPL 关节顺序 -> MuJoCo/PHC 关节顺序
   - optional upright_start
   - optional on_the_ground
2. 对变换后的局部旋转做 SkeletonState FK，得到全局关节坐标。
3. 与
   output/HumanoidIm/phc_comp_3/phc_act/
   phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl
   中的 pos_state/gt_pos_state（或 pred_pos/gt_pred_pos）比较。

默认参数按目标文件名设置为：
  --process_split test --upright_start --gender_flag --on_the_ground

示例：
  python scripts/data_process/validate_3dpw_pose_fk_vs_jointpositions.py

  python scripts/data_process/validate_3dpw_pose_fk_vs_jointpositions.py ^
    --phc_act_pkl output/HumanoidIm/phc_comp_3/phc_act/phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl ^
    --output data/3dpw/fk_vs_phc_act_report.json
"""

import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


R_3DPW_TO_AMASS = sRot.from_euler("x", 90, degrees=True)
Q_UPRIGHT_ALIGN = sRot.from_quat([0.5, 0.5, 0.5, 0.5])


def to_numpy(x: Any) -> np.ndarray:
    """Convert numpy / torch / list-like value to numpy."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def gender_to_convert_v2(gender: Any, use_gender: bool) -> Tuple[str, Iterable[int], np.ndarray]:
    """
    Match convert_3dpw_data_v2.py gender behavior.

    When --gender_flag is disabled, convert_3dpw_data_v2.py forces neutral robot
    but uses gender_number=[2].  The target file name contains "gender", so the
    default path normally uses the enabled branch below.
    """
    beta = np.zeros(16, dtype=np.float32)
    if not use_gender:
        return "neutral", [2], beta

    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ("male", "m"):
        return "male", [1], beta
    if gender in ("female", "f"):
        return "female", [2], beta
    return "neutral", [0], beta


def smpl_to_mujoco_indices() -> list:
    """Indices that convert SMPL_BONE_ORDER_NAMES order to SMPL_MUJOCO_NAMES order."""
    return [SMPL_BONE_ORDER_NAMES.index(name) for name in SMPL_MUJOCO_NAMES if name in SMPL_BONE_ORDER_NAMES]


def make_robot(upright_start: bool) -> LocalRobot:
    """Robot config copied from convert_3dpw_data_v2.py."""
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
    return LocalRobot(robot_cfg)


def make_skeleton_tree(
    robot: LocalRobot,
    beta: np.ndarray,
    gender_number: Iterable[int],
    tmp_xml_path: str,
) -> SkeletonTree:
    os.makedirs(osp.dirname(tmp_xml_path), exist_ok=True)
    robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]).float(),
        gender=gender_number,
        objs_info=None,
    )
    robot.write_xml(tmp_xml_path)
    return SkeletonTree.from_mjcf(tmp_xml_path)


def build_pose_quat_and_root_trans(pose_aa_full: np.ndarray, root_trans_3dpw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Copy convert_3dpw_data_v2.py rotation/translation conversion exactly.

    - root_trans_amass = Rx(+90deg).apply(root_trans)
    - root_rot_amass = Rx(+90deg) * root_rot_cam
    - pose_aa = concat(first 66 pose dims, zeros(6))
    - pose_aa_mj = pose_aa[:, smpl_2_mujoco]
    """
    n = pose_aa_full.shape[0]
    root_trans_amass = R_3DPW_TO_AMASS.apply(root_trans_3dpw[:n])

    pose_aa_amass = pose_aa_full[:n].copy()
    root_rot = sRot.from_rotvec(pose_aa_amass[:, :3])
    pose_aa_amass[:, :3] = (R_3DPW_TO_AMASS * root_rot).as_rotvec()

    pose_aa = np.concatenate([pose_aa_amass[:, :66], np.zeros((n, 6), dtype=pose_aa_amass.dtype)], axis=-1)
    pose_aa_mj = pose_aa.reshape(n, 24, 3)[:, smpl_to_mujoco_indices()]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(n, 24, 4)
    return pose_quat, root_trans_amass


def fk_like_convert_v2(
    pose_aa_full: np.ndarray,
    root_trans_3dpw: np.ndarray,
    beta: np.ndarray,
    gender_number: Iterable[int],
    robot: LocalRobot,
    tmp_xml_path: str,
    upright_start: bool,
    on_the_ground: bool,
) -> Dict[str, np.ndarray]:
    """Run the same SkeletonState FK pipeline as convert_3dpw_data_v2.py."""
    pose_quat, root_trans_amass = build_pose_quat_and_root_trans(pose_aa_full, root_trans_3dpw)
    n = pose_quat.shape[0]

    skeleton_tree = make_skeleton_tree(robot, beta, gender_number, tmp_xml_path)
    root_trans_offset = torch.from_numpy(root_trans_amass).float() + skeleton_tree.local_translation[0]

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat).float(),
        root_trans_offset,
        is_local=True,
    )

    if upright_start:
        pose_quat_global = (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).detach().cpu().numpy())
            * Q_UPRIGHT_ALIGN.inv()
        ).as_quat().reshape(n, 24, 4)
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global).float(),
            root_trans_offset,
            is_local=False,
        )

    floor_offset = np.zeros(n, dtype=np.float64)
    if on_the_ground:
        up_axis_idx = 2 if upright_start else 1
        global_joints = sk_state.global_translation
        min_heights, _ = torch.min(global_joints[..., up_axis_idx], dim=1)
        root_trans_offset_fixed = root_trans_offset.clone()
        root_trans_offset_fixed[:, up_axis_idx] -= min_heights

        # This is intentionally the same call as convert_3dpw_data_v2.py:
        # keep local_rotation and rebuild from the ground-aligned root.
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            sk_state.local_rotation,
            root_trans_offset_fixed,
            is_local=True,
        )
        root_trans_offset = root_trans_offset_fixed
        floor_offset = min_heights.detach().cpu().numpy()

    return {
        "fk_pos_mujoco": sk_state.global_translation.detach().cpu().numpy(),
        "pose_quat_global": sk_state.global_rotation.detach().cpu().numpy(),
        "pose_quat": sk_state.local_rotation.detach().cpu().numpy(),
        "root_trans_offset": root_trans_offset.detach().cpu().numpy(),
        "floor_offset": floor_offset,
    }


def first_existing(data: Dict[str, Any], aliases: Iterable[str]) -> Tuple[Any, str]:
    for name in aliases:
        if name in data and data[name] is not None:
            return data[name], name
    return None, ""


def flatten_key_names(key_names: Any) -> list:
    """Flatten PHC key_names while preserving each motion key as a string."""
    arr = np.asarray(key_names, dtype=object)
    out = []
    for item in arr.reshape(-1):
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_key_names(item))
        else:
            out.append(str(item))
    return out


def load_phc_positions(action_pkl: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    """
    Load target PHC action positions.

    Supports both:
      - pos_state / gt_pos_state
      - pred_pos / gt_pred_pos

    Return canonical field names:
      positions["pos_state"][motion_key] and positions["gt_pos_state"][motion_key]
    """
    data = joblib.load(action_pkl)
    if not isinstance(data, dict):
        raise TypeError(f"{action_pkl} should contain a dict, got {type(data)}")

    key_names = data.get("key_names")
    if key_names is None:
        raise KeyError(f"{action_pkl} does not contain key_names")
    key_names = flatten_key_names(key_names)

    aliases = {
        "pos_state": ("pos_state", "pred_pos"),
        "gt_pos_state": ("gt_pos_state", "gt_pred_pos"),
    }

    positions: Dict[str, Dict[str, np.ndarray]] = {}
    source_fields: Dict[str, str] = {}
    for canonical, names in aliases.items():
        values, source_name = first_existing(data, names)
        if values is None:
            positions[canonical] = {}
            continue
        source_fields[canonical] = source_name

        if isinstance(values, dict):
            positions[canonical] = {str(k): to_numpy(v) for k, v in values.items()}
        else:
            positions[canonical] = {
                key: to_numpy(values[idx])
                for idx, key in enumerate(key_names)
                if idx < len(values)
            }

    return positions, source_fields


def compare_positions(fk_pos: np.ndarray, target_pos: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    """Compare two (N,24,3) arrays in the same MuJoCo/PHC joint order."""
    fk_pos = np.asarray(fk_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if fk_pos.ndim == 2:
        fk_pos = fk_pos.reshape(fk_pos.shape[0], 24, 3)
    if target_pos.ndim == 2:
        target_pos = target_pos.reshape(target_pos.shape[0], 24, 3)
    if fk_pos.ndim != 3 or fk_pos.shape[1:] != (24, 3):
        raise ValueError(f"FK positions should be (N,24,3), got {fk_pos.shape}")
    if target_pos.ndim != 3 or target_pos.shape[1:] != (24, 3):
        raise ValueError(f"Target positions should be (N,24,3), got {target_pos.shape}")

    n = min(fk_pos.shape[0], target_pos.shape[0])
    a = fk_pos[:n]
    b = target_pos[:n]
    diff = a - b
    joint_l2 = np.linalg.norm(diff, axis=-1)
    per_joint_mean_l2 = joint_l2.mean(axis=0) if joint_l2.size else np.zeros(24)
    worst_joint = int(np.argmax(per_joint_mean_l2)) if per_joint_mean_l2.size else -1

    return {
        "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "num_frames_compared": int(n),
        "fk_num_frames": int(fk_pos.shape[0]),
        "target_num_frames": int(target_pos.shape[0]),
        "max_abs_coord_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_coord_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse_coord": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "max_joint_l2_error": float(np.max(joint_l2)) if joint_l2.size else 0.0,
        "mean_joint_l2_error": float(np.mean(joint_l2)) if joint_l2.size else 0.0,
        "worst_joint_index_mujoco_order": worst_joint,
        "worst_joint_name": SMPL_MUJOCO_NAMES[worst_joint] if worst_joint >= 0 else "",
        "worst_joint_mean_l2_error": float(per_joint_mean_l2[worst_joint]) if worst_joint >= 0 else 0.0,
    }


def summarize(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not reports:
        return {"num_motions": 0, "num_allclose": 0, "all_motions_allclose": False}
    return {
        "num_motions": len(reports),
        "num_allclose": int(sum(r["allclose"] for r in reports.values())),
        "all_motions_allclose": bool(all(r["allclose"] for r in reports.values())),
        "max_abs_coord_error": float(max(r["max_abs_coord_error"] for r in reports.values())),
        "mean_abs_coord_error": float(np.mean([r["mean_abs_coord_error"] for r in reports.values()])),
        "rmse_coord": float(np.sqrt(np.mean([r["rmse_coord"] ** 2 for r in reports.values()]))),
        "max_joint_l2_error": float(max(r["max_joint_l2_error"] for r in reports.values())),
        "mean_joint_l2_error": float(np.mean([r["mean_joint_l2_error"] for r in reports.values()])),
    }


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    phc_positions, phc_source_fields = load_phc_positions(args.phc_act_pkl)
    pkl_paths = sorted(glob.glob(osp.join(args.path, "sequence", args.process_split, "*.pkl")))
    if not pkl_paths:
        raise FileNotFoundError(f"No 3DPW pkl files found under {osp.join(args.path, 'sequence', args.process_split)}")

    robot = make_robot(args.upright_start)
    reports = {"pos_state": {}, "gt_pos_state": {}}
    missing = {"pos_state": [], "gt_pos_state": []}

    for pkl_path in tqdm(pkl_paths, desc="3DPW convert-v2 FK -> compare PHC action pkl"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        seq_name = osp.basename(pkl_path).replace(".pkl", "")
        num_people = len(data["poses"])
        if args.skip_single_person and num_people <= 1:
            continue

        for person_id in range(num_people):
            key = f"{seq_name}_{person_id}"
            pose_aa_full = np.asarray(data["poses"][person_id], dtype=np.float64)
            root_trans = np.asarray(data["trans"][person_id], dtype=np.float64)
            n = min(pose_aa_full.shape[0], root_trans.shape[0])
            if n < args.min_frames:
                continue

            gender_str, gender_number, beta = gender_to_convert_v2(data["genders"][person_id], args.gender_flag)
            if args.gender_flag:
                beta[:10] = np.asarray(data["betas"][person_id][:10], dtype=np.float32)

            tmp_xml_path = osp.join(args.tmp_xml_dir, f"smpl_fk_validate_{os.getpid()}_{person_id}.xml")
            fk = fk_like_convert_v2(
                pose_aa_full=pose_aa_full[:n],
                root_trans_3dpw=root_trans[:n],
                beta=beta,
                gender_number=gender_number,
                robot=robot,
                tmp_xml_path=tmp_xml_path,
                upright_start=args.upright_start,
                on_the_ground=args.on_the_ground,
            )
            fk_pos = fk["fk_pos_mujoco"]

            for field in ("pos_state", "gt_pos_state"):
                if key not in phc_positions.get(field, {}):
                    if phc_positions.get(field):
                        missing[field].append(key)
                    continue
                report = compare_positions(fk_pos, phc_positions[field][key], args.atol, args.rtol)
                report.update(
                    {
                        "sequence": seq_name,
                        "person_id": int(person_id),
                        "gender": gender_str,
                        "beta_used": bool(args.gender_flag),
                        "upright_start": bool(args.upright_start),
                        "on_the_ground": bool(args.on_the_ground),
                        "floor_offset_min": float(np.min(fk["floor_offset"])) if len(fk["floor_offset"]) else 0.0,
                        "floor_offset_max": float(np.max(fk["floor_offset"])) if len(fk["floor_offset"]) else 0.0,
                        "floor_offset_mean": float(np.mean(fk["floor_offset"])) if len(fk["floor_offset"]) else 0.0,
                    }
                )
                reports[field][key] = report

    result = {
        "description": (
            "FK result from 3DPW poses after reproducing convert_3dpw_data_v2.py "
            "coordinate transforms, compared directly with PHC action positions in MuJoCo joint order."
        ),
        "path": args.path,
        "process_split": args.process_split,
        "phc_act_pkl": args.phc_act_pkl,
        "phc_source_fields": phc_source_fields,
        "joint_order": "MuJoCo/PHC (SMPL_MUJOCO_NAMES)",
        "coordinate_transform": "root_trans_amass=Rx(+90deg)*trans; root_rot_amass=Rx(+90deg)*root_rot",
        "upright_start": bool(args.upright_start),
        "on_the_ground": bool(args.on_the_ground),
        "gender_flag": bool(args.gender_flag),
        "atol": args.atol,
        "rtol": args.rtol,
        "summary": {field: summarize(field_reports) for field, field_reports in reports.items()},
        "motions": reports,
        "missing_phc_keys": {k: sorted(set(v)) for k, v in missing.items() if v},
    }

    if args.output:
        output_dir = osp.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {args.output}")

    for field, summary in result["summary"].items():
        print(
            f"{field}: allclose {summary['num_allclose']}/{summary['num_motions']}, "
            f"max_abs={summary.get('max_abs_coord_error', 0.0):.6g}, "
            f"max_joint_l2={summary.get('max_joint_l2_error', 0.0):.6g}, "
            f"mean_joint_l2={summary.get('mean_joint_l2_error', 0.0):.6g}, "
            f"rmse={summary.get('rmse_coord', 0.0):.6g}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/3dpw/Vanilla")
    parser.add_argument("--process_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--phc_act_pkl",
        type=str,
        default="output/HumanoidIm/phc_comp_3/phc_act/phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl",
    )
    parser.add_argument("--output", type=str, default="data/3dpw/fk_vs_phc_act_report.json")
    parser.add_argument("--tmp_xml_dir", type=str, default="phc/data/assets/mjcf")

    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument("--no_upright_start", dest="upright_start", action="store_false")
    parser.add_argument("--on_the_ground", dest="on_the_ground", action="store_true", default=True)
    parser.add_argument("--no_on_the_ground", dest="on_the_ground", action="store_false")
    parser.add_argument("--gender_flag", dest="gender_flag", action="store_true", default=True)
    parser.add_argument("--no_gender_flag", dest="gender_flag", action="store_false")

    parser.add_argument("--skip_single_person", dest="skip_single_person", action="store_true", default=True)
    parser.add_argument("--include_single_person", dest="skip_single_person", action="store_false")
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
