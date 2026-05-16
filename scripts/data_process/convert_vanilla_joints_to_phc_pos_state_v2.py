import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys
import warnings

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())

try:
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
except Exception as exc:  # pragma: no cover - only used when PHC deps are not installed
    # Keep a local copy of the PHC/SMPLSim names so this script can still do the
    # exact convert_3dpw_data_v2.py reorder in lightweight data-processing envs.
    SMPL_BONE_ORDER_NAMES = [
        "Pelvis",
        "L_Hip",
        "R_Hip",
        "Torso",
        "L_Knee",
        "R_Knee",
        "Spine",
        "L_Ankle",
        "R_Ankle",
        "Chest",
        "L_Toe",
        "R_Toe",
        "Neck",
        "L_Thorax",
        "R_Thorax",
        "Head",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Hand",
        "R_Hand",
    ]
    SMPL_MUJOCO_NAMES = [
        "Pelvis",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Toe",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Toe",
        "Torso",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "L_Thorax",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Hand",
        "R_Thorax",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Hand",
    ]
    warnings.warn(
        f"Cannot import smpl_sim joint names ({exc}); using embedded PHC/SMPLSim SMPL joint names.",
        RuntimeWarning,
    )


R_3DPW_TO_PHC = sRot.from_euler("x", 90, degrees=True)
R_PHC_TO_3DPW = R_3DPW_TO_PHC.inv()
Q_UPRIGHT_ALIGN = sRot.from_quat([0.5, 0.5, 0.5, 0.5])


def smpl_mujoco_maps(num_joints=24):
    """Return the same SMPL <-> MuJoCo joint-order maps used by convert_3dpw_data_v2.py."""
    smpl_2_mujoco = np.asarray(
        [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES],
        dtype=np.int64,
    )
    if len(smpl_2_mujoco) != num_joints:
        raise ValueError(f"Expected {num_joints} SMPL/MuJoCo joints, got map of length {len(smpl_2_mujoco)}")
    mujoco_2_smpl = np.argsort(smpl_2_mujoco)
    return smpl_2_mujoco, mujoco_2_smpl


def to_numpy(x):
    """Accept numpy arrays and torch tensors without requiring torch at import time."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_gender(gender):
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ["male", "m"]:
        return "male"
    if gender in ["female", "f"]:
        return "female"
    return "neutral"


def get_beta(data, person_id, gender_flag=True):
    if gender_flag:
        beta = np.zeros(16, dtype=np.float32)
        beta[:10] = np.asarray(data["betas"][person_id][:10], dtype=np.float32)
        return beta
    return np.zeros(16, dtype=np.float32)


def normalize_joint_positions(joints):
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 2:
        if joints.shape[1] != 72:
            raise ValueError(f"Expected flattened jointPositions shape (N, 72), got {joints.shape}")
        joints = joints.reshape(joints.shape[0], 24, 3)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected jointPositions shape (N, 24, 3) or (N, 72), got {joints.shape}")
    return joints


def vanilla_joints_to_phc(
    joints_3dpw_cam,
    cam_poses,
    root_trans_3dpw_cam,
    upright_start=True,
    on_the_ground=True,
    root_local_translation=None,
):
    """
    Forward coordinate transform for 3DPW jointPositions, matching convert_3dpw_data_v2.py.

    Input joints are the Vanilla 3DPW `jointPositions` in original camera coordinates and SMPL
    joint order.  Output is PHC/AMASS coordinates in MuJoCo joint order, suitable as `pos_state`.

    The transform mirrors the rotation-vector script's coordinate operations:
      1. camera -> 3DPW world: p_world = R_cam @ p_cam + t_cam
      2. 3DPW -> PHC/AMASS: rotate +90 degrees about X
      3. SMPL joint order -> MuJoCo joint order
      4. if upright_start: rotate points around root with Q_UPRIGHT_ALIGN.inv()
      5. if on_the_ground: subtract each frame's minimum height from the PHC up axis
    """
    joints_3dpw_cam = normalize_joint_positions(joints_3dpw_cam)
    cam_poses = np.asarray(cam_poses, dtype=np.float64)
    root_trans_3dpw_cam = np.asarray(root_trans_3dpw_cam, dtype=np.float64)

    n = joints_3dpw_cam.shape[0]
    cam_poses = cam_poses[:n]
    root_trans_3dpw_cam = root_trans_3dpw_cam[:n]
    r_cam = cam_poses[:, :3, :3]
    t_cam = cam_poses[:, :3, 3]

    # Same root translation conversion as convert_3dpw_data_v2.py.
    root_world = np.einsum("nij,nj->ni", r_cam, root_trans_3dpw_cam) + t_cam
    trans_orig = R_3DPW_TO_PHC.apply(root_world)

    if root_local_translation is None:
        # In the PHC SMPL MJCF the root body's local translation is normally zero.  The
        # parameter is kept so callers can pass skeleton_tree.local_translation[0] if needed.
        root_local_translation = np.zeros(3, dtype=np.float64)
    root_local_translation = np.asarray(root_local_translation, dtype=np.float64)
    root_trans_offset = trans_orig + root_local_translation[None, :]

    # Full point transform: camera -> world -> PHC/AMASS.
    joints_world = np.einsum("nij,nkj->nki", r_cam, joints_3dpw_cam) + t_cam[:, None, :]
    joints_phc = R_3DPW_TO_PHC.apply(joints_world.reshape(-1, 3)).reshape(n, -1, 3)

    # SMPL order -> MuJoCo order.
    smpl_2_mujoco, _ = smpl_mujoco_maps(joints_phc.shape[1])
    joints_phc = joints_phc[:, smpl_2_mujoco, :]

    # Match convert_3dpw_data_v2.py's upright global-rotation alignment in point space.
    if upright_start:
        centered = joints_phc - trans_orig[:, None, :]
        joints_phc = Q_UPRIGHT_ALIGN.inv().apply(centered.reshape(-1, 3)).reshape(n, -1, 3)
        joints_phc = joints_phc + root_trans_offset[:, None, :]
    else:
        joints_phc = joints_phc + (root_trans_offset - trans_orig)[:, None, :]

    floor_offset = np.zeros(n, dtype=np.float64)
    if on_the_ground:
        up_axis_idx = 2 if upright_start else 1
        floor_offset = np.min(joints_phc[..., up_axis_idx], axis=1)
        joints_phc = joints_phc.copy()
        joints_phc[..., up_axis_idx] -= floor_offset[:, None]
        root_trans_offset = root_trans_offset.copy()
        root_trans_offset[:, up_axis_idx] -= floor_offset

    return joints_phc, trans_orig, root_trans_offset, floor_offset


def phc_joints_to_vanilla(
    joints_phc,
    cam_poses,
    trans_orig,
    root_trans_offset,
    upright_start=True,
    to_camera=True,
):
    """
    Inverse of `vanilla_joints_to_phc`.

    Returns joints in SMPL order.  With `to_camera=True` (default), coordinates are restored to
    the original Vanilla 3DPW camera coordinate system.
    """
    joints_phc = np.asarray(joints_phc, dtype=np.float64)
    cam_poses = np.asarray(cam_poses, dtype=np.float64)
    trans_orig = np.asarray(trans_orig, dtype=np.float64)
    root_trans_offset = np.asarray(root_trans_offset, dtype=np.float64)
    n = joints_phc.shape[0]

    if upright_start:
        centered = joints_phc - root_trans_offset[:, None, :]
        joints_3dpw_order = Q_UPRIGHT_ALIGN.apply(centered.reshape(-1, 3)).reshape(n, -1, 3)
        joints_3dpw_order = joints_3dpw_order + trans_orig[:, None, :]
    else:
        joints_3dpw_order = joints_phc + (trans_orig - root_trans_offset)[:, None, :]

    # PHC/AMASS -> 3DPW world.
    joints_3dpw_order = R_PHC_TO_3DPW.apply(joints_3dpw_order.reshape(-1, 3)).reshape(n, -1, 3)

    # MuJoCo order -> SMPL order.
    _, mujoco_2_smpl = smpl_mujoco_maps(joints_3dpw_order.shape[1])
    joints_3dpw_order = joints_3dpw_order[:, mujoco_2_smpl, :]

    if to_camera:
        cam_poses = cam_poses[:n]
        r_cam = cam_poses[:, :3, :3]
        t_cam = cam_poses[:, :3, 3]
        joints_3dpw_order = np.einsum(
            "nij,nkj->nki",
            np.transpose(r_cam, (0, 2, 1)),
            joints_3dpw_order - t_cam[:, None, :],
        )

    return joints_3dpw_order


def compare_arrays(restored, target, atol=1e-5, rtol=1e-5):
    n = min(restored.shape[0], target.shape[0])
    diff = np.asarray(restored[:n], dtype=np.float64) - np.asarray(target[:n], dtype=np.float64)
    joint_l2 = np.linalg.norm(diff, axis=-1)
    return {
        "num_frames_compared": int(n),
        "allclose": bool(np.allclose(restored[:n], target[:n], atol=atol, rtol=rtol)),
        "max_abs_coord_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_coord_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse_coord": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "max_joint_l2_error": float(np.max(joint_l2)) if joint_l2.size else 0.0,
        "mean_joint_l2_error": float(np.mean(joint_l2)) if joint_l2.size else 0.0,
    }


def summarize_reports(reports):
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


def process_3dpw_joint_positions(args):
    data_path = osp.join(args.path, "sequence", args.process_split)
    all_pkls = sorted(glob.glob(osp.join(data_path, "*.pkl")))
    output = {}
    validation = {
        "source": args.path,
        "split": args.process_split,
        "atol": args.atol,
        "rtol": args.rtol,
        "motions": {},
    }

    for pkl_path in tqdm(all_pkls, desc="3DPW jointPositions -> PHC pos_state"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        seq_name = osp.basename(pkl_path).replace(".pkl", "")
        num_people = len(data["jointPositions"])
        if args.skip_single_person and num_people <= 1:
            continue

        for person_id in range(num_people):
            key_name = f"{seq_name}_{person_id}"
            joints_cam = normalize_joint_positions(data["jointPositions"][person_id])
            root_trans = np.asarray(data["trans"][person_id], dtype=np.float64)
            n = min(joints_cam.shape[0], root_trans.shape[0], data["cam_poses"].shape[0])
            if n < args.min_frames:
                continue

            joints_cam = joints_cam[:n]
            root_trans = root_trans[:n]
            cam_poses = np.asarray(data["cam_poses"][:n], dtype=np.float64)

            joints_phc, trans_orig, root_trans_offset, floor_offset = vanilla_joints_to_phc(
                joints_cam,
                cam_poses,
                root_trans,
                upright_start=args.upright_start,
                on_the_ground=args.on_the_ground,
            )

            restored = phc_joints_to_vanilla(
                joints_phc,
                cam_poses,
                trans_orig,
                root_trans_offset,
                upright_start=args.upright_start,
                to_camera=True,
            )
            validation["motions"][key_name] = compare_arrays(restored, joints_cam, args.atol, args.rtol)

            gender = get_gender(data["genders"][person_id]) if args.gender_flag else "neutral"
            output[key_name] = {
                "pos_state": joints_phc.astype(np.float32),
                "joint_positions_phc": joints_phc.astype(np.float32),
                "trans_orig": trans_orig.astype(np.float32),
                "root_trans_offset": root_trans_offset.astype(np.float32),
                "floor_offset": floor_offset.astype(np.float32),
                "cam_poses": cam_poses.astype(np.float32),
                "beta": get_beta(data, person_id, args.gender_flag),
                "gender": gender,
                "fps": 30,
                "source_joint_field": "jointPositions",
            }

    validation["summary"] = summarize_reports(validation["motions"])

    output_dir = osp.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(output, args.output, compress=True)
    print(f"Saved PHC joint-position data to {args.output}")

    print(
        "Round-trip validation: "
        f"allclose {validation['summary']['num_allclose']}/{validation['summary']['num_motions']}, "
        f"max_abs={validation['summary'].get('max_abs_coord_error', 0.0):.6g}, "
        f"max_joint_l2={validation['summary'].get('max_joint_l2_error', 0.0):.6g}, "
        f"rmse={validation['summary'].get('rmse_coord', 0.0):.6g}"
    )

    if args.validation_output:
        validation_dir = osp.dirname(args.validation_output)
        if validation_dir:
            os.makedirs(validation_dir, exist_ok=True)
        with open(args.validation_output, "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, ensure_ascii=False)
        print(f"Saved validation report to {args.validation_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert Vanilla 3DPW jointPositions to PHC pos_state coordinates using the same "
            "coordinate transforms as convert_3dpw_data_v2.py, then inverse-transform to validate."
        )
    )
    parser.add_argument("--path", type=str, default="data/3dpw/Vanilla")
    parser.add_argument("--process_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default="data/3dpw/3dpw_test_jointpos_upright_whole_sequence_gender_ground.pkl")
    parser.add_argument("--validation_output", type=str, default="data/3dpw/jointpos_phc_roundtrip_validation.json")
    parser.add_argument("--upright_start", dest="upright_start", action="store_true", default=True)
    parser.add_argument("--no_upright_start", dest="upright_start", action="store_false")
    parser.add_argument("--gender_flag", action="store_true", default=True)
    parser.add_argument("--on_the_ground", action="store_true", default=True)
    parser.add_argument("--skip_single_person", action="store_true", default=True)
    parser.add_argument("--include_single_person", dest="skip_single_person", action="store_false")
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    process_3dpw_joint_positions(args)
