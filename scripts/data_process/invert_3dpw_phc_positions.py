"""
Invert the coordinate transform used by convert_3dpw_data_v2.py for PHC
spatial joint positions, convert PHC/MuJoCo joint order back to SMPL order,
then compare with the original 3DPW Vanilla pkl files.

Input PHC action file fields:
  - pred_pos
  - gt_pred_pos

Both fields are expected to be object arrays indexed by key_names.  Each item is
an (T, 24, 3) array in PHC/MuJoCo joint order and in the converted
AMASS/upright/ground coordinate system.

Default example:
  python scripts/data_process/invert_3dpw_phc_positions.py

Outputs:
  - data/3dpw/invert_3dpw_phc_positions_report.json
  - data/3dpw/inverted_phc_act_3dpw_positions.pkl

Notes:
  convert_3dpw_data_v2.py stores the per-frame floor offset and transformed
  root translation in data/3dpw/3dpw_test_upright_whole_sequence_gender_ground.pkl.
  This script uses that converted pkl to undo the ground shift and to recover a
  3DPW-space root anchor for absolute-coordinate comparison.
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
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


R_AMASS_TO_3DPW = sRot.from_euler("x", -90, degrees=True)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def smpl_to_mujoco_indices() -> list:
    """Indices used by convert_3dpw_data_v2.py: SMPL order -> PHC/MuJoCo order."""
    return [SMPL_BONE_ORDER_NAMES.index(name) for name in SMPL_MUJOCO_NAMES if name in SMPL_BONE_ORDER_NAMES]


def mujoco_to_smpl_indices() -> list:
    """
    Indices that reorder a PHC/MuJoCo-order array back to SMPL_BONE_ORDER_NAMES.

    If convert_3dpw_data_v2.py did:
        pose_mujoco = pose_smpl[:, smpl_to_mujoco_indices()]
    then this script does:
        pose_smpl = pose_mujoco[:, mujoco_to_smpl_indices()]
    """
    smpl_to_mj = smpl_to_mujoco_indices()
    if sorted(smpl_to_mj) != list(range(len(smpl_to_mj))):
        raise ValueError("SMPL/MuJoCo mapping is not a full permutation; cannot invert it safely.")
    return np.argsort(smpl_to_mj).tolist()


def reorder_mujoco_to_smpl(pos_mujoco: np.ndarray) -> np.ndarray:
    """Reorder last joint axis from PHC/MuJoCo order to Vanilla SMPL order."""
    return np.asarray(pos_mujoco)[:, mujoco_to_smpl_indices()]


def flatten_key_names(key_names: Any) -> list:
    arr = np.asarray(key_names, dtype=object)
    out = []
    for item in arr.reshape(-1):
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_key_names(item))
        else:
            out.append(str(item))
    return out


def load_action_data(
    action_pkl: str,
    fields: Iterable[str],
    rotation_fields: Iterable[str],
) -> Tuple[list, Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    data = joblib.load(action_pkl)
    if "key_names" not in data:
        raise KeyError(f"{action_pkl} does not contain key_names")

    key_names = flatten_key_names(data["key_names"])
    positions: Dict[str, Dict[str, np.ndarray]] = {}
    for field in fields:
        if field not in data:
            print(f"[warn] {action_pkl} does not contain {field}; skip it")
            continue
        values = np.asarray(data[field], dtype=object)
        if len(values) != len(key_names):
            raise ValueError(f"{field} length {len(values)} != key_names length {len(key_names)}")
        positions[field] = {key: np.asarray(values[i]) for i, key in enumerate(key_names)}

    rotations: Dict[str, Dict[str, np.ndarray]] = {}
    for field in rotation_fields:
        if field not in data:
            continue
        values = np.asarray(data[field], dtype=object)
        if len(values) != len(key_names):
            raise ValueError(f"{field} length {len(values)} != key_names length {len(key_names)}")
        rotations[field] = {key: np.asarray(values[i]) for i, key in enumerate(key_names)}
    return key_names, positions, rotations


def load_vanilla_joint_positions(vanilla_dir: str) -> Dict[str, np.ndarray]:
    """Load Vanilla/sequence/test/*.pkl jointPositions and keep the original SMPL order."""
    targets: Dict[str, np.ndarray] = {}
    pkl_paths = sorted(glob.glob(osp.join(vanilla_dir, "*.pkl")))
    if not pkl_paths:
        raise FileNotFoundError(f"No Vanilla pkl files found under {vanilla_dir}")

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        seq_name = osp.basename(pkl_path).replace(".pkl", "")
        if "jointPositions" not in data:
            continue
        for person_id, joints_flat in enumerate(data["jointPositions"]):
            joints = np.asarray(joints_flat, dtype=np.float64).reshape(-1, 24, 3)
            targets[f"{seq_name}_{person_id}"] = joints
    return targets


def recover_original_root_anchor(entry: Dict[str, Any]) -> np.ndarray:
    """
    Estimate Vanilla 3DPW root-joint position from convert_3dpw_data_v2 output.

    In convert_3dpw_data_v2.py:
      trans_orig        = Rx(+90) * original trans
      root_trans_offset = trans_orig + skeleton_tree.local_translation[0]
      root_trans_offset[:, z] -= floor_offset

    Empirically, for this PHC/SMPL robot the saved root local offset is in the
    same component order needed to anchor Vanilla jointPositions.  Therefore:
      original_trans = Rx(-90) * trans_orig
      root_pre_ground = root_trans_offset with floor_offset added back on z
      local_root_offset = root_pre_ground - trans_orig
      original_root_anchor = original_trans + local_root_offset
    """
    trans_orig = to_numpy(entry["trans_orig"]).astype(np.float64)
    root_trans_offset = to_numpy(entry["root_trans_offset"]).astype(np.float64)
    floor_offset = to_numpy(entry.get("floor_offset", np.zeros(len(trans_orig)))).astype(np.float64).reshape(-1)

    n = min(len(trans_orig), len(root_trans_offset), len(floor_offset))
    trans_orig = trans_orig[:n]
    root_pre_ground = root_trans_offset[:n].copy()
    root_pre_ground[:, 2] += floor_offset[:n]

    original_trans = R_AMASS_TO_3DPW.apply(trans_orig)
    local_root_offset = root_pre_ground - trans_orig
    return original_trans + local_root_offset


def make_robot() -> LocalRobot:
    """Robot config copied from convert_3dpw_data_v2.py."""
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
    return LocalRobot(robot_cfg)


def gender_number_from_entry(entry: Dict[str, Any]) -> list:
    gender = str(entry.get("gender", "neutral")).lower()
    if gender in ("male", "m"):
        return [1]
    if gender in ("female", "f"):
        return [2]
    return [0]


def make_skeleton_tree(entry: Dict[str, Any], tmp_xml_path: str, robot: LocalRobot) -> SkeletonTree:
    os.makedirs(osp.dirname(tmp_xml_path), exist_ok=True)
    beta = to_numpy(entry.get("beta", np.zeros(16))).astype(np.float32)
    robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]).float(),
        gender=gender_number_from_entry(entry),
        objs_info=None,
    )
    robot.write_xml(tmp_xml_path)
    return SkeletonTree.from_mjcf(tmp_xml_path)


def undo_upright_start_with_global_rot(
    pos_amass_pre_ground: np.ndarray,
    global_rot_after_upright: np.ndarray,
    skeleton_tree: SkeletonTree,
) -> np.ndarray:
    """
    Undo convert_3dpw_data_v2.py's upright_start for positions.

    convert_3dpw_data_v2.py does not rotate joint coordinates directly.  It
    changes global rotations and then recomputes FK with the same root
    translation.  Therefore the spatial-coordinate counterpart is not a single
    global rotation of the point cloud.  Given the post-upright global rotations
    and the same skeleton offsets, recover the corresponding FK positions edge
    by edge:

        p_pre[root] = p_after[root]
        p_pre[j]    = p_pre[parent] + H_parent.apply(local_offset[j])
    """
    pos = np.asarray(pos_amass_pre_ground, dtype=np.float64)
    rot = np.asarray(global_rot_after_upright, dtype=np.float64)
    n = min(len(pos), len(rot))
    pos = pos[:n]
    rot = rot[:n]

    parents = skeleton_tree.parent_indices.detach().cpu().numpy().astype(int)
    local_translation = skeleton_tree.local_translation.detach().cpu().numpy().astype(np.float64)

    out = np.zeros_like(pos)
    out[:, 0, :] = pos[:, 0, :]
    for joint_id in range(1, len(parents)):
        parent_id = int(parents[joint_id])
        parent_rot_pre = sRot.from_quat(rot[:, parent_id, :])
        offset = np.broadcast_to(local_translation[joint_id], (n, 3))
        out[:, joint_id, :] = out[:, parent_id, :] + parent_rot_pre.apply(offset)
    return out


def inverse_position_transform(
    pos_amass_ground: np.ndarray,
    converted_entry: Dict[str, Any],
    align_root_anchor: bool = True,
    upright_start: bool = True,
    global_rot_after_upright: np.ndarray = None,
    skeleton_tree: SkeletonTree = None,
) -> np.ndarray:
    """
    Invert the spatial-coordinate parts of convert_3dpw_data_v2.py:
      1) undo on_the_ground: z += floor_offset
      2) if upright_start is enabled, undo the FK-level upright rotation using
         final global rotations and the skeleton tree
      3) undo 3DPW -> AMASS: Rx(-90deg)
      4) optional: shift every frame so joint 0 matches the recovered Vanilla
         root anchor from the converted pkl.

    The optional root anchoring is useful because PHC positions are robot FK
    joint positions, while Vanilla jointPositions use the original 3DPW SMPL
    joint root.  Without anchoring, relative pose can still be compared with
    root-aligned metrics.
    """
    pos = np.asarray(pos_amass_ground, dtype=np.float64).copy()
    if pos.ndim != 3 or pos.shape[-2:] != (24, 3):
        raise ValueError(f"Expected position shape (T, 24, 3), got {pos.shape}")

    floor_offset = to_numpy(converted_entry.get("floor_offset", np.zeros(len(pos)))).astype(np.float64).reshape(-1)
    n = min(len(pos), len(floor_offset))
    pos = pos[:n]
    pos[:, :, 2] += floor_offset[:n, None]

    if upright_start:
        if global_rot_after_upright is None or skeleton_tree is None:
            raise ValueError(
                "upright_start=True requires global_rot_after_upright and skeleton_tree; "
                "use --no_upright_start to skip this inverse step."
            )
        pos = undo_upright_start_with_global_rot(pos, global_rot_after_upright, skeleton_tree)

    pos = R_AMASS_TO_3DPW.apply(pos.reshape(-1, 3)).reshape(pos.shape)

    if align_root_anchor:
        root_anchor = recover_original_root_anchor(converted_entry)
        n = min(len(pos), len(root_anchor))
        pos = pos[:n]
        pos += root_anchor[:n, None, :] - pos[:, :1, :]
    return pos


def compare_positions(source: np.ndarray, target: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    n = min(len(source), len(target))
    source = source[:n]
    target = target[:n]
    diff = source - target
    joint_l2 = np.linalg.norm(diff, axis=-1)
    return {
        "num_frames_compared": int(n),
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "allclose": bool(np.allclose(source, target, atol=atol, rtol=rtol)),
        "max_abs_coord_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_coord_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse_coord": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "max_joint_l2_error": float(np.max(joint_l2)) if joint_l2.size else 0.0,
        "mean_joint_l2_error": float(np.mean(joint_l2)) if joint_l2.size else 0.0,
    }


def root_center(pos: np.ndarray) -> np.ndarray:
    return pos - pos[:, :1, :]


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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    key_names, action_positions, action_rotations = load_action_data(args.phc_act_pkl, args.fields, args.rotation_fields)
    converted = joblib.load(args.converted_motion_pkl)
    vanilla = load_vanilla_joint_positions(args.vanilla_dir)
    robot = make_robot()

    inverted_output: Dict[str, Dict[str, np.ndarray]] = {field: {} for field in action_positions}
    reports = {
        field: {"raw_inverse": {}, "root_anchor_aligned": {}, "root_centered": {}}
        for field in action_positions
    }
    missing = {"converted": [], "vanilla": []}

    for key in tqdm(key_names, desc="inverse PHC positions -> compare Vanilla jointPositions"):
        if key not in converted:
            missing["converted"].append(key)
            continue
        if key not in vanilla:
            missing["vanilla"].append(key)
            continue

        target = vanilla[key]
        skeleton_tree = None
        if args.upright_start:
            tmp_xml_path = osp.join(args.tmp_xml_dir, f"smpl_invert_pos_{os.getpid()}_{key}.xml")
            skeleton_tree = make_skeleton_tree(converted[key], tmp_xml_path, robot)

        for field, field_positions in action_positions.items():
            if key not in field_positions:
                continue

            rot_field = args.position_to_rotation_field.get(field, "")
            if args.upright_start:
                if rot_field == "__converted__":
                    global_rot = to_numpy(converted[key]["pose_quat_global"])
                elif rot_field and rot_field in action_rotations and key in action_rotations[rot_field]:
                    global_rot = action_rotations[rot_field][key]
                else:
                    # Fallback is mostly for gt_pred_pos when the action pkl
                    # lacks gt_rot.  It uses the converted dataset's saved
                    # post-upright global rotations.
                    global_rot = to_numpy(converted[key]["pose_quat_global"])
            else:
                global_rot = None

            raw_inv = inverse_position_transform(
                field_positions[key],
                converted[key],
                align_root_anchor=False,
                upright_start=args.upright_start,
                global_rot_after_upright=global_rot,
                skeleton_tree=skeleton_tree,
            )
            anchored_inv = inverse_position_transform(
                field_positions[key],
                converted[key],
                align_root_anchor=True,
                upright_start=args.upright_start,
                global_rot_after_upright=global_rot,
                skeleton_tree=skeleton_tree,
            )
            raw_inv_smpl = reorder_mujoco_to_smpl(raw_inv)
            anchored_inv_smpl = reorder_mujoco_to_smpl(anchored_inv)

            inverted_output[field][key] = anchored_inv_smpl.astype(np.float32)
            
            reports[field]["raw_inverse"][key] = compare_positions(raw_inv_smpl, target, args.atol, args.rtol)
            reports[field]["root_anchor_aligned"][key] = compare_positions(anchored_inv_smpl, target, args.atol, args.rtol)
            reports[field]["root_centered"][key] = compare_positions(
                root_center(raw_inv_smpl), root_center(target), args.atol, args.rtol
            )

    result = {
        "description": (
            "pred_pos/gt_pred_pos are inverse-transformed from PHC converted coordinates "
            "back to 3DPW coordinates, reordered from MuJoCo/PHC joint order back to "
            "SMPL joint order, and compared with Vanilla jointPositions. Vanilla "
            "jointPositions are kept in their original SMPL order."
        ),
        "phc_act_pkl": args.phc_act_pkl,
        "converted_motion_pkl": args.converted_motion_pkl,
        "vanilla_dir": args.vanilla_dir,
        "fields": list(action_positions.keys()),
        "joint_order": "SMPL_BONE_ORDER_NAMES",
        "phc_input_joint_order": "MuJoCo/PHC (SMPL_MUJOCO_NAMES)",
        "vanilla_joint_order": "Original Vanilla SMPL order (unchanged)",
        "inverse_transform": [
            "undo ground: z += floor_offset",
            f"undo upright_start with global rotations and skeleton offsets: {bool(args.upright_start)}",
            "undo coordinate conversion: Rx(-90deg)",
            "root_anchor_aligned additionally shifts joint 0 to the recovered 3DPW root anchor",
            "reorder PHC/MuJoCo joints back to SMPL_BONE_ORDER_NAMES; Vanilla joints are not reordered",
        ],
        "position_to_rotation_field": args.position_to_rotation_field,
        "summary": {
            field: {mode: summarize(mode_reports) for mode, mode_reports in field_reports.items()}
            for field, field_reports in reports.items()
        },
        "motions": reports,
        "missing": {k: sorted(set(v)) for k, v in missing.items() if v},
    }

    if args.output_report:
        os.makedirs(osp.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {args.output_report}")

    if args.output_pkl:
        os.makedirs(osp.dirname(args.output_pkl) or ".", exist_ok=True)
        save_data = {
            'rl_pos': inverted_output['pred_pos'],
            'rl_gt_pos': inverted_output['gt_pred_pos']
        }
        joblib.dump(
            {
                "description": "Root-anchor-aligned inverse 3DPW-space PHC positions in SMPL joint order.",
                "key_names": np.asarray(key_names),
                "fields": save_data,
            },
            args.output_pkl,
            compress=True,
        )
        print(f"Saved inverted positions to {args.output_pkl}")

    print("\n=== inverse-position comparison summary ===")
    for field, mode_summaries in result["summary"].items():
        for mode, summary in mode_summaries.items():
            print(
                f"{field}/{mode}: allclose {summary['num_allclose']}/{summary['num_motions']}, "
                f"max_abs={summary.get('max_abs_coord_error', 0.0):.6g}, "
                f"max_joint_l2={summary.get('max_joint_l2_error', 0.0):.6g}, "
                f"mean_joint_l2={summary.get('mean_joint_l2_error', 0.0):.6g}, "
                f"rmse={summary.get('rmse_coord', 0.0):.6g}"
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phc_act_pkl",
        type=str,
        default="output/HumanoidIm/phc_comp_3/phc_act/phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl",
    )
    parser.add_argument(
        "--converted_motion_pkl",
        type=str,
        default="data/3dpw/3dpw_test_upright_whole_sequence_gender_ground.pkl",
    )
    parser.add_argument("--vanilla_dir", type=str, default="data/3dpw/Vanilla/sequence/test")
    parser.add_argument("--output_report", type=str, default="data/3dpw/invert_3dpw_phc_positions_report.json")
    parser.add_argument("--output_pkl", type=str, default="data/3dpw/data_offline_multi3dpw.pkl")
    parser.add_argument("--tmp_xml_dir", type=str, default="phc/data/assets/mjcf")
    parser.add_argument(
        "--fields",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["pred_pos", "gt_pred_pos"],
        help="Comma-separated fields in the PHC action pkl to invert and compare.",
    )
    parser.add_argument(
        "--rotation_fields",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["pred_rot", "gt_rot"],
        help="Comma-separated global-rotation fields used to undo upright_start.",
    )
    parser.add_argument(
        "--position_to_rotation_field",
        type=lambda s: dict(item.split(":", 1) for item in s.split(",") if item.strip()),
        default={"pred_pos": "pred_rot", "gt_pred_pos": "__converted__"},
        help=(
            "Mapping like pred_pos:pred_rot,gt_pred_pos:gt_rot. "
            "Use __converted__ to use pose_quat_global from --converted_motion_pkl."
        ),
    )
    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument("--no_upright_start", dest="upright_start", action="store_false")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
