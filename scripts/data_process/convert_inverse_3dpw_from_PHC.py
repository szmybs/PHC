import argparse
import os
import sys
import warnings

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES


R_AMASS_TO_3DPW = sRot.from_euler("x", -90, degrees=True)
Q_UPRIGHT_ALIGN = sRot.from_quat([0.5, 0.5, 0.5, 0.5])


def to_numpy(x):
    """Convert numpy / torch tensor / list-like data to a numpy array."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def smpl_to_mujoco_indices():
    return [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]


def mujoco_to_smpl_indices():
    return np.argsort(smpl_to_mujoco_indices())


def load_action_positions(action_path):
    """
    Load PHC action rollout positions.

    The target file stores:
      - pred_pos:    list/array aligned with key_names, MuJoCo joint order
      - gt_pred_pos: list/array aligned with key_names, MuJoCo joint order

    Some older files may use pos_state / gt_pos_state or dict fields; keep those
    aliases so the output format remains compatible with the original script.
    """
    data = joblib.load(action_path)
    if "key_names" not in data:
        raise KeyError(f"{action_path} does not contain 'key_names'.")
    key_names = [str(k) for k in data["key_names"]]

    def build_map(*aliases):
        values = None
        source_name = None
        for name in aliases:
            if name in data and data[name] is not None:
                values = data[name]
                source_name = name
                break
        if values is None:
            return {}, None
        if isinstance(values, dict):
            return {str(k): to_numpy(v) for k, v in values.items()}, source_name
        return {
            key: to_numpy(values[i])
            for i, key in enumerate(key_names)
            if i < len(values)
        }, source_name

    pred, pred_source = build_map("pred_pos", "pos_state")
    gt_pred, gt_source = build_map("gt_pred_pos", "gt_pos_state")
    print(f"Loaded PHC positions from {action_path}: pred={pred_source}, gt={gt_source}")
    return pred, gt_pred


def inverse_conversion(val, pos_key, args):
    """
    Invert the coordinate transforms that convert_3dpw_data_v2.py applies before
    saving PHC-ready data.

    Important: convert_3dpw_data_v2.py transforms rotations with FK, while this
    function transforms already-materialized joint coordinates (pred_pos /
    gt_pred_pos).  Therefore the inverse is applied directly to points:

      final PHC coords
        -> undo on_the_ground translation
        -> undo upright_start point rotation around root_trans_offset
        -> AMASS/PHC coords -> 3DPW coords by Rx(-90deg)
        -> MuJoCo joint order -> SMPL joint order

    The current forward script does not apply cam_poses to root translation or
    root orientation, so cam_poses is intentionally not used here.
    """
    joints = to_numpy(val[pos_key]).astype(np.float64, copy=True)
    if joints.ndim == 2:
        joints = joints.reshape(joints.shape[0], 24, 3)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"{pos_key} should have shape (N, 24, 3), got {joints.shape}")

    n = joints.shape[0]
    root_trans_offset = to_numpy(val["root_trans_offset"]).astype(np.float64)[:n].copy()

    # convert_3dpw_data_v2.py shifts the whole skeleton by subtracting the
    # per-frame minimum height from root_trans_offset on the up axis.  Undo this
    # first for coordinates and for the upright rotation center.
    if args.on_the_ground:
        up_axis_idx = 2 if args.upright_start else 1
        floor_offset = to_numpy(val.get("floor_offset", np.zeros(n))).astype(np.float64)[:n]
        joints[..., up_axis_idx] += floor_offset[:, None]
        root_trans_offset[:, up_axis_idx] += floor_offset

    # convert_3dpw_data_v2.py applies global_rotation * Q_UPRIGHT_ALIGN.inv().
    # For coordinates this is a point rotation by Q_UPRIGHT_ALIGN.inv() around
    # root_trans_offset; invert it with Q_UPRIGHT_ALIGN.
    if args.upright_start:
        centered = joints - root_trans_offset[:, None, :]
        joints = (
            Q_UPRIGHT_ALIGN.apply(centered.reshape(-1, 3)).reshape(n, 24, 3)
            + root_trans_offset[:, None, :]
        )

    # AMASS/PHC coordinate system back to 3DPW coordinate system.  This is the
    # inverse of R_3dpw_2_amass = Rx(+90deg) in convert_3dpw_data_v2.py.
    joints = R_AMASS_TO_3DPW.apply(joints.reshape(-1, 3)).reshape(n, 24, 3)

    # PHC positions are in MuJoCo order; downstream 3DPW code expects SMPL order,
    # matching the original convert_inverse_3dpw_from_PHC.py output.
    joints = joints[:, mujoco_to_smpl_indices(), :]
    return joints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phc_stat_name",
        type=str,
        default="data/3dpw/3dpw_test_upright_whole_sequence_gender_ground.pkl",
    )
    parser.add_argument(
        "--phc_act_stat_name",
        type=str,
        default="output/HumanoidIm/phc_comp_3/phc_act/phc_act_3dpw_test_upright_whole_sequence_gender_ground.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "3dpw", "data_offline_multi3dpw.pkl"),
    )
    parser.add_argument("--gender_flag", action="store_true", default=True)
    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument("--no_upright_start", dest="upright_start", action="store_false")
    parser.add_argument("--on_the_ground", dest="on_the_ground", action="store_true", default=True)
    parser.add_argument("--no_on_the_ground", dest="on_the_ground", action="store_false")
    args = parser.parse_args()

    phc_act_stat, phc_gt_act_stat = load_action_positions(args.phc_act_stat_name)
    phc_stat_full = joblib.load(args.phc_stat_name)

    for key, value in phc_act_stat.items():
        if key in phc_stat_full:
            phc_stat_full[key]["pos_state"] = value
            if key in phc_gt_act_stat:
                phc_stat_full[key]["gt_pos_state"] = phc_gt_act_stat[key]
        else:
            warnings.warn(f"{key} does not exist in {args.phc_stat_name}", RuntimeWarning)

    restored_motions, restored_gt_motions = {}, {}
    phc_motions, phc_gt_motions = {}, {}
    for key, val in tqdm(phc_stat_full.items(), desc="Inverse convert PHC coordinates"):
        if "pos_state" in val:
            restored_motions[key] = inverse_conversion(val, "pos_state", args)
            phc_motions[key] = val["pos_state"][:, mujoco_to_smpl_indices(), :]
        if "gt_pos_state" in val:
            restored_gt_motions[key] = inverse_conversion(val, "gt_pos_state", args)
            phc_gt_motions[key] = val["gt_pos_state"][:, mujoco_to_smpl_indices(), :]

    data = {
        "rl_pos": restored_motions,
        "rl_gt_pos": restored_gt_motions,
        "rl_pos_phc": phc_motions,
        "rl_gt_pos_phc": phc_gt_motions
    }
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(data, output_path)
    print(f"Coordinate inverse conversion finished. Saved to {output_path}")


if __name__ == "__main__":
    main()
