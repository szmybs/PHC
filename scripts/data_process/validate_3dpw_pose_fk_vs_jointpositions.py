import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys

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


def gender_to_number(gender):
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ["male", "m"]:
        return "male", [1]
    if gender in ["female", "f"]:
        return "female", [2]
    return "neutral", [0]


def smpl_to_mujoco_indices():
    return [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]


def mujoco_to_smpl_indices():
    return np.argsort(smpl_to_mujoco_indices())


def normalize_joints(joints):
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 2:
        joints = joints.reshape(joints.shape[0], 24, 3)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"Expected jointPositions shape (N, 72) or (N, 24, 3), got {joints.shape}")
    return joints


def to_numpy(x):
    """Accept numpy arrays / torch tensors from PHC joblib files."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def first_existing(data, names):
    for name in names:
        if name in data and data[name] is not None:
            return data[name], name
    return None, None


def load_phc_action_data(action_path):
    """
    Load PHC rollout positions and return {field: {motion_key: (N, 24, 3)}}.

    Supports both naming conventions:
      - pos_state / gt_pos_state
      - pred_pos / gt_pred_pos
    """
    if not action_path:
        return {}, {}
    action_data = joblib.load(action_path)
    key_names = action_data.get("key_names")
    if key_names is None:
        raise KeyError(f"{action_path} does not contain 'key_names'.")
    key_names = [str(k) for k in key_names]

    field_aliases = {
        "pos_state": ["pos_state", "pred_pos"],
        "gt_pos_state": ["gt_pos_state", "gt_pred_pos"],
    }
    mapped = {}
    source_names = {}
    for canonical_name, aliases in field_aliases.items():
        values, source_name = first_existing(action_data, aliases)
        if values is None:
            continue
        source_names[canonical_name] = source_name
        if isinstance(values, dict):
            mapped[canonical_name] = {str(k): to_numpy(v) for k, v in values.items()}
        else:
            mapped[canonical_name] = {
                key: to_numpy(values[idx])
                for idx, key in enumerate(key_names)
                if idx < len(values)
            }
    return mapped, source_names


def make_robot(robot_upright_start=False):
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": robot_upright_start,
        "remove_toe": False,
        "real_weight": True,
        "replace_feet": True,
        "big_ankle": True,
        "model": "smpl",
        "body_params": {},
    }
    return LocalRobot(robot_cfg)


def make_skeleton_tree(robot, beta, gender_number, tmp_xml_path):
    os.makedirs(osp.dirname(tmp_xml_path), exist_ok=True)
    robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]).float(),
        gender=gender_number,
        objs_info=None,
    )
    robot.write_xml(tmp_xml_path)
    return SkeletonTree.from_mjcf(tmp_xml_path)


def build_pose_quat_and_root(
    pose_aa_full,
    trans,
    cam_poses=None,
    zero_last_two_hands=False,
    convert_3dpw_to_amass=True,
):
    """
    Build local quaternions in the same frame as `trans`.

    If `cam_poses` is given, convert the SMPL root/global orientation from
    camera coordinates to 3DPW world coordinates:

        R_root_world = R_cam @ R_root_cam

    If `convert_3dpw_to_amass` is True, then apply the same 3DPW->AMASS transform
    as convert_3dpw_data_v2.py:

        root_trans_amass = R_3DPW_TO_AMASS @ root_trans_world
        R_root_amass = R_3DPW_TO_AMASS @ R_root_world

    Child joint rotations are local rotations, so they are not affected by the
    camera/world or 3DPW/AMASS frame changes.
    """
    n = pose_aa_full.shape[0]
    pose_aa_24 = pose_aa_full.copy()
    root_trans = trans[:n]

    if cam_poses is not None:
        r_cam = np.asarray(cam_poses[:n, :3, :3], dtype=np.float64)
        root_rot_cam = sRot.from_rotvec(pose_aa_24[:, :3])
        root_rot_world = sRot.from_matrix(r_cam) * root_rot_cam
        pose_aa_24[:, :3] = root_rot_world.as_rotvec()

    if convert_3dpw_to_amass:
        root_trans = R_3DPW_TO_AMASS.apply(root_trans)
        root_rot_world = sRot.from_rotvec(pose_aa_24[:, :3])
        pose_aa_24[:, :3] = (R_3DPW_TO_AMASS * root_rot_world).as_rotvec()

    if zero_last_two_hands:
        pose_aa_24[:, 66:72] = 0.0
    pose_aa_mj = pose_aa_24.reshape(n, 24, 3)[:, smpl_to_mujoco_indices()]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(n, 24, 4)
    return pose_quat, root_trans


def camera_to_world_points(points_cam, cam_poses):
    """Apply Vanilla 3DPW camera -> world transform: p_world = R_cam @ p_cam + t_cam."""
    n = points_cam.shape[0]
    r_cam = cam_poses[:n, :3, :3]
    t_cam = cam_poses[:n, :3, 3]
    return np.einsum("nij,nkj->nki", r_cam, points_cam) + t_cam[:, None, :]


def camera_to_world_trans(trans_cam, cam_poses):
    """Apply Vanilla 3DPW camera -> world transform to root translations."""
    n = trans_cam.shape[0]
    r_cam = cam_poses[:n, :3, :3]
    t_cam = cam_poses[:n, :3, 3]
    return np.einsum("nij,nj->ni", r_cam, trans_cam) + t_cam


def points_3dpw_to_amass(points_3dpw):
    """Apply the same 3DPW -> AMASS coordinate transform as convert_3dpw_data_v2.py."""
    points_3dpw = np.asarray(points_3dpw, dtype=np.float64)
    return R_3DPW_TO_AMASS.apply(points_3dpw.reshape(-1, 3)).reshape(points_3dpw.shape)


def apply_upright_to_points(points_amass, upright_center):
    """
    Apply the same upright alignment used in convert_3dpw_data_v2.py to points.

    convert_3dpw_data_v2.py applies:
        global_rot_upright = global_rot * Q_UPRIGHT_ALIGN.inv()

    For point coordinates this is equivalent to rotating each joint around the root
    by Q_UPRIGHT_ALIGN.inv().
    """
    points_amass = np.asarray(points_amass, dtype=np.float64)
    upright_center = np.asarray(upright_center, dtype=np.float64)
    n = points_amass.shape[0]
    centered = points_amass - upright_center[:, None, :]
    return Q_UPRIGHT_ALIGN.inv().apply(centered.reshape(-1, 3)).reshape(n, -1, 3) + upright_center[:, None, :]


def apply_ground_correction_to_points(points, floor_offset, up_axis_idx):
    """Subtract per-frame floor_offset on the up axis, matching convert_3dpw_data_v2.py."""
    points = np.asarray(points, dtype=np.float64).copy()
    floor_offset = np.asarray(floor_offset, dtype=np.float64)
    points[..., up_axis_idx] -= floor_offset[:, None]
    return points


def compare(fk_joints, target_joints, atol, rtol):
    fk_joints = np.asarray(fk_joints, dtype=np.float64)
    target_joints = np.asarray(target_joints, dtype=np.float64)
    n = min(fk_joints.shape[0], target_joints.shape[0])
    fk_cmp = fk_joints[:n]
    target_cmp = target_joints[:n]
    diff = fk_cmp - target_cmp
    joint_l2 = np.linalg.norm(diff, axis=-1)
    per_joint_mean_l2 = np.mean(joint_l2, axis=0)
    worst_joint_idx = int(np.argmax(per_joint_mean_l2)) if per_joint_mean_l2.size else -1
    return {
        "allclose": bool(np.allclose(fk_cmp, target_cmp, atol=atol, rtol=rtol)),
        "num_frames_compared": int(n),
        "left_num_frames": int(fk_joints.shape[0]),
        "right_num_frames": int(target_joints.shape[0]),
        "max_abs_coord_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_coord_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse_coord": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "max_joint_l2_error": float(np.max(joint_l2)) if joint_l2.size else 0.0,
        "mean_joint_l2_error": float(np.mean(joint_l2)) if joint_l2.size else 0.0,
        "worst_joint_index_smpl_order": worst_joint_idx,
        "worst_joint_name": SMPL_BONE_ORDER_NAMES[worst_joint_idx] if worst_joint_idx >= 0 else "",
        "worst_joint_mean_l2_error": float(per_joint_mean_l2[worst_joint_idx]) if worst_joint_idx >= 0 else 0.0,
    }


def summarize(reports):
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


def validate(args):
    data_path = osp.join(args.path, "sequence", args.process_split)
    pkl_paths = sorted(glob.glob(osp.join(data_path, "*.pkl")))
    robot = make_robot(robot_upright_start=args.robot_upright_start)
    phc_action_data, phc_action_source_names = load_phc_action_data(args.phc_act_stat_name)

    reports = {}
    phc_comparison_reports = {
        "pos_state_vs_jointpositions": {},
        "pos_state_vs_poses_fk": {},
        "gt_pos_state_vs_jointpositions": {},
        "gt_pos_state_vs_poses_fk": {},
    }
    missing_phc_keys = {"pos_state": [], "gt_pos_state": []}
    for pkl_path in tqdm(pkl_paths, desc="Validate 3DPW poses FK vs jointPositions"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        seq_name = osp.basename(pkl_path).replace(".pkl", "")
        num_people = len(data["poses"])
        if args.skip_single_person and num_people <= 1:
            continue

        for person_id in range(num_people):
            key = f"{seq_name}_{person_id}"
            pose_aa_full = np.asarray(data["poses"][person_id], dtype=np.float64)
            trans = np.asarray(data["trans"][person_id], dtype=np.float64)
            joints_cam = normalize_joints(data["jointPositions"][person_id])
            cam_poses = np.asarray(data["cam_poses"], dtype=np.float64)
            n = min(pose_aa_full.shape[0], trans.shape[0], joints_cam.shape[0], cam_poses.shape[0])
            if n < args.min_frames:
                continue
            pose_aa_full = pose_aa_full[:n]
            trans = trans[:n]
            joints_cam = joints_cam[:n]
            cam_poses = cam_poses[:n]

            trans_world = camera_to_world_trans(trans, cam_poses)
            joints_world = camera_to_world_points(joints_cam, cam_poses)
            joints_amass = points_3dpw_to_amass(joints_world)

            gender_str, gender_number = gender_to_number(data["genders"][person_id])
            if not args.gender_flag:
                gender_str, gender_number = "neutral", [0]
                beta = np.zeros(16, dtype=np.float32)
            else:
                beta = np.zeros(16, dtype=np.float32)
                beta[:10] = np.asarray(data["betas"][person_id][:10], dtype=np.float32)

            pose_quat, root_trans = build_pose_quat_and_root(
                pose_aa_full,
                trans_world,
                cam_poses=cam_poses,
                zero_last_two_hands=args.zero_last_two_hands,
                convert_3dpw_to_amass=True,
            )

            tmp_xml_path = osp.join(args.tmp_xml_dir, f"smpl_fk_validate_{os.getpid()}_{person_id}.xml")
            skeleton_tree = make_skeleton_tree(robot, beta, gender_number, tmp_xml_path)
            root_trans_offset = torch.from_numpy(root_trans).float() + skeleton_tree.local_translation[0]

            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat).float(),
                root_trans_offset,
                is_local=True,
            )

            if args.upright_start:
                pose_quat_global_upright = (
                    sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).detach().cpu().numpy())
                    * Q_UPRIGHT_ALIGN.inv()
                ).as_quat().reshape(n, 24, 4)
                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat_global_upright).float(),
                    root_trans_offset,
                    is_local=False,
                )
                root_trans_offset_np = root_trans_offset.detach().cpu().numpy()
                if args.upright_center == "root_trans_offset":
                    upright_center = root_trans_offset_np
                elif args.upright_center == "joint0":
                    upright_center = joints_amass[:, 0, :]
                else:
                    raise ValueError(f"Unsupported upright_center: {args.upright_center}")

                joints_target = apply_upright_to_points(
                    joints_amass,
                    upright_center,
                )
            else:
                root_trans_offset_np = root_trans_offset.detach().cpu().numpy()
                upright_center = None
                joints_target = joints_amass

            floor_offset = np.zeros(n, dtype=np.float64)
            if args.on_the_ground:
                up_axis_idx = 2 if args.upright_start else 1
                fk_joints_before_ground = sk_state.global_translation.detach().cpu().numpy()
                floor_offset = np.min(fk_joints_before_ground[..., up_axis_idx], axis=1)

                root_trans_offset_fixed = root_trans_offset.clone()
                root_trans_offset_fixed[:, up_axis_idx] -= torch.from_numpy(floor_offset).float()
                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    sk_state.local_rotation,
                    root_trans_offset_fixed,
                    is_local=True,
                )
                joints_target = apply_ground_correction_to_points(
                    joints_target,
                    floor_offset,
                    up_axis_idx,
                )

            # SkeletonState outputs MuJoCo order. Reorder only the joint order back to
            # SMPL. Both arrays are now in AMASS/PHC coordinates.
            fk_joints_mujoco = sk_state.global_translation.detach().cpu().numpy()
            fk_joints_smpl = fk_joints_mujoco[:, mujoco_to_smpl_indices(), :]
            joints_target_mujoco = joints_target[:, smpl_to_mujoco_indices(), :]

            report = compare(fk_joints_smpl, joints_target, args.atol, args.rtol)
            pre_upright_root_diff = joints_amass[:, 0, :] - root_trans_offset_np
            report.update(
                {
                    "gender": gender_str,
                    "beta_used": bool(args.gender_flag),
                    "upright_center": args.upright_center if args.upright_start else "none",
                    "on_the_ground": bool(args.on_the_ground),
                    "ground_up_axis_idx": int(2 if args.upright_start else 1),
                    "floor_offset_min": float(np.min(floor_offset)) if floor_offset.size else 0.0,
                    "floor_offset_max": float(np.max(floor_offset)) if floor_offset.size else 0.0,
                    "floor_offset_mean": float(np.mean(floor_offset)) if floor_offset.size else 0.0,
                    "pre_upright_joint0_vs_root_trans_offset_max_abs": float(np.max(np.abs(pre_upright_root_diff))),
                    "pre_upright_joint0_vs_root_trans_offset_mean_l2": float(
                        np.mean(np.linalg.norm(pre_upright_root_diff, axis=-1))
                    ),
                }
            )
            reports[key] = report

            for phc_field in ["pos_state", "gt_pos_state"]:
                field_data = phc_action_data.get(phc_field, {})
                if key not in field_data:
                    if phc_field in phc_action_data:
                        missing_phc_keys[phc_field].append(key)
                    continue

                phc_positions = to_numpy(field_data[key]).astype(np.float64)
                if phc_positions.ndim != 3 or phc_positions.shape[1:] != (24, 3):
                    raise ValueError(
                        f"{phc_field}[{key}] should have shape (N, 24, 3), got {phc_positions.shape}"
                    )

                phc_comparison_reports[f"{phc_field}_vs_jointpositions"][key] = compare(
                    phc_positions,
                    joints_target_mujoco,
                    args.atol,
                    args.rtol,
                )
                phc_comparison_reports[f"{phc_field}_vs_poses_fk"][key] = compare(
                    phc_positions,
                    fk_joints_mujoco,
                    args.atol,
                    args.rtol,
                )

    result = {
        "description": "Compare FK from Vanilla 3DPW poses against Vanilla jointPositions after applying Camera->world, 3DPW->AMASS, optional upright alignment, and optional ground correction to root translation, root orientation, and jointPositions.",
        "path": args.path,
        "split": args.process_split,
        "coordinate_frame": "AMASS/PHC coordinates",
        "camera_to_world_translation": "p_world = R_cam @ p_cam + t_cam",
        "camera_to_world_root_orientation": "R_root_world = R_cam @ R_root_cam",
        "3dpw_to_amass_translation": "p_amass = R_3DPW_TO_AMASS @ p_world, R_3DPW_TO_AMASS = Rx(+90deg)",
        "3dpw_to_amass_root_orientation": "R_root_amass = R_3DPW_TO_AMASS @ R_root_world",
        "upright_start": args.upright_start,
        "upright_center": args.upright_center if args.upright_start else "none",
        "upright_alignment": "global_rot_upright = global_rot * Q_UPRIGHT_ALIGN.inv(); target points are rotated around the selected upright_center by Q_UPRIGHT_ALIGN.inv()",
        "on_the_ground": args.on_the_ground,
        "ground_correction": "floor_offset is min FK joint height per frame on up_axis_idx=(2 if upright_start else 1); FK root and target points subtract floor_offset on that axis",
        "robot_upright_start": args.robot_upright_start,
        "zero_last_two_hands": args.zero_last_two_hands,
        "gender_flag": args.gender_flag,
        "phc_act_stat_name": args.phc_act_stat_name,
        "phc_action_source_fields": phc_action_source_names,
        "atol": args.atol,
        "rtol": args.rtol,
        "summary": summarize(reports),
        "motions": reports,
        "phc_action_comparisons": {
            name: {
                "summary": summarize(motion_reports),
                "motions": motion_reports,
            }
            for name, motion_reports in phc_comparison_reports.items()
        },
        "missing_phc_action_keys": {
            field: sorted(set(keys))
            for field, keys in missing_phc_keys.items()
            if keys
        },
    }

    if args.output:
        output_dir = osp.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved validation report to {args.output}")

    s = result["summary"]
    print(
        f"FK vs jointPositions: allclose {s['num_allclose']}/{s['num_motions']}, "
        f"max_abs={s.get('max_abs_coord_error', 0.0):.6g}, "
        f"max_joint_l2={s.get('max_joint_l2_error', 0.0):.6g}, "
        f"mean_joint_l2={s.get('mean_joint_l2_error', 0.0):.6g}, "
        f"rmse={s.get('rmse_coord', 0.0):.6g}"
    )
    for name, comp in result["phc_action_comparisons"].items():
        cs = comp["summary"]
        print(
            f"{name}: allclose {cs['num_allclose']}/{cs['num_motions']}, "
            f"max_abs={cs.get('max_abs_coord_error', 0.0):.6g}, "
            f"max_joint_l2={cs.get('max_joint_l2_error', 0.0):.6g}, "
            f"mean_joint_l2={cs.get('mean_joint_l2_error', 0.0):.6g}, "
            f"rmse={cs.get('rmse_coord', 0.0):.6g}"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether 3DPW Vanilla poses produce jointPositions through PHC "
            "SkeletonState FK after applying Camera->world, 3DPW->AMASS, upright alignment, and ground correction."
        )
    )
    parser.add_argument("--path", type=str, default="data/3dpw/Vanilla")
    parser.add_argument("--process_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default="data/3dpw/pose_fk_vs_jointpositions_validation.json")
    parser.add_argument("--tmp_xml_dir", type=str, default="phc/data/assets/mjcf")
    parser.add_argument(
        "--phc_act_stat_name",
        type=str,
        default="sample_data/phc_act/3dpw_test_upright_whole_sequence_gender_ground/noise_False_0.05.pkl",
        help=(
            "PHC rollout pkl containing key_names and either pos_state/gt_pos_state "
            "or pred_pos/gt_pred_pos. These positions are expected in MuJoCo joint order."
        ),
    )
    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument("--no_upright_start", dest="upright_start", action="store_false")
    parser.add_argument("--on_the_ground", action="store_true", default=True)
    parser.add_argument("--no_on_the_ground", dest="on_the_ground", action="store_false")
    parser.add_argument(
        "--upright_center",
        type=str,
        default="root_trans_offset",
        choices=["root_trans_offset", "joint0"],
        help=(
            "Rotation center used when applying upright to Vanilla jointPositions. "
            "'root_trans_offset' matches convert_3dpw_data_v2.py FK; 'joint0' rotates around jointPositions pelvis."
        ),
    )
    parser.add_argument(
        "--robot_upright_start",
        action="store_true",
        default=True,
        help="Only affects SMPL_Robot skeleton construction. Default matches convert_3dpw_data_v2.py.",
    )
    parser.add_argument("--no_robot_upright_start", dest="robot_upright_start", action="store_false")
    parser.add_argument(
        "--zero_last_two_hands",
        action="store_true",
        default=True,
        help="Set the last two SMPL hand joint rotations to zero before FK, matching convert_3dpw_data_v2.py.",
    )
    parser.add_argument("--gender_flag", action="store_true", default=False)
    parser.add_argument("--skip_single_person", action="store_true", default=True)
    parser.add_argument("--include_single_person", dest="skip_single_person", action="store_false")
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    validate(parser.parse_args())
