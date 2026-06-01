#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline evaluation for starVLA QwenGR00T galbot bimanual NO-STATE policy.

Same as GalbotBimanualRelative/offline_eval but:
  - No state input (include_state=false, state_dim=0)
  - Added trajectory error report (cm) at steps 10, 20, 30

Comparison is done in **unnormalized delta space** (physical units):
  - GT  = raw abs action chunk from LeRobot, then transformed to delta:
          * pos: arithmetic difference (abs[t] - abs[t-1], first=0)
          * rotation_6d: SO(3) relative rotation (R[t] @ R[t-1]^T, first=I)
          * gripper: binary threshold on abs values (>100mm → 1)
  - Pred = model normalized_actions → inverse-normalized back to delta physical space

Minimal CLI:
  python offline_eval_galbot_no_state.py \
      --checkpoint   /abs/path/to/steps_80000_pytorch_model.pt \
      --dataset-path /abs/path/to/galbot_lerobot_dual_cup_0526 \
      --episode-idx 0 --frame-idx 0

Action layout (20D):
  [0:3] left_pos | [3:9] left_ori_6d | [9] left_gripper |
  [10:13] right_pos | [13:19] right_ori_6d | [19] right_gripper
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.dataloader.gr00t_lerobot.transform.rotation_utils import (
    compute_relative_rotation_rot6d,
    identity_rotation_rot6d,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_YAML = (
    _REPO_ROOT
    / "examples/GalbotBimanualNoState/train_files/starvla_qwengroot_galbot_no_state.yaml"
)

ACTION_DIM = 20
ACTION_NAMES = (
    [f"L_pos_{i}" for i in "xyz"]
    + [f"L_ori6d_{i}" for i in range(6)]
    + ["L_gripper"]
    + [f"R_pos_{i}" for i in "xyz"]
    + [f"R_ori6d_{i}" for i in range(6)]
    + ["R_gripper"]
)
LEFT_POS_SLICE  = slice(0, 3)
LEFT_ORI_SLICE  = slice(3, 9)
LEFT_GRIP_IDX   = 9
RIGHT_POS_SLICE = slice(10, 13)
RIGHT_ORI_SLICE = slice(13, 19)
RIGHT_GRIP_IDX  = 19

LEFT_DIMS  = list(range(0, 10))
RIGHT_DIMS = list(range(10, 20))
ARM_SPECS  = [("left", LEFT_DIMS), ("right", RIGHT_DIMS)]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("offline_eval_galbot_no_state")


# ---------------------------------------------------------------------------
# Stats / (de)normalization helpers
# ---------------------------------------------------------------------------
def _load_stats(dataset_path: Path) -> dict:
    p = dataset_path / "meta" / "stats.json"
    with open(p) as f:
        return json.load(f)["statistics"]


def _denorm_q99(normed: np.ndarray, q01: np.ndarray, q99: np.ndarray,
                col_indices: list[int]) -> np.ndarray:
    out = normed.copy()
    for i in col_indices:
        rng = q99[i] - q01[i]
        if rng < 1e-8:
            pass
        else:
            out[:, i] = (normed[:, i] + 1.0) / 2.0 * rng + q01[i]
    return out


def normalized_to_delta(normed: np.ndarray, stats: dict) -> np.ndarray:
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
    pos_cols = list(range(0, 3)) + list(range(10, 13))
    return _denorm_q99(normed.astype(np.float32), q01, q99, pos_cols)


def abs_to_delta(abs_chunk: np.ndarray) -> np.ndarray:
    """Convert [T, 20] abs action chunk → delta.

    - pos: arithmetic difference (delta[t] = abs[t] - abs[t-1], delta[0] = 0)
    - rotation_6d: SO(3) relative rotation (R[t] @ R[t-1]^T, R[0] = I)
    - gripper: kept as-is (will be binarized separately)
    """
    delta = np.zeros_like(abs_chunk, dtype=np.float32)

    if abs_chunk.shape[0] > 1:
        # pos: arithmetic difference
        delta[1:, LEFT_POS_SLICE] = abs_chunk[1:, LEFT_POS_SLICE] - abs_chunk[:-1, LEFT_POS_SLICE]
        delta[1:, RIGHT_POS_SLICE] = abs_chunk[1:, RIGHT_POS_SLICE] - abs_chunk[:-1, RIGHT_POS_SLICE]

        # rotation_6d: SO(3) relative rotation
        for t in range(1, abs_chunk.shape[0]):
            delta[t, LEFT_ORI_SLICE] = compute_relative_rotation_rot6d(
                abs_chunk[t, LEFT_ORI_SLICE], abs_chunk[t - 1, LEFT_ORI_SLICE]
            )
            delta[t, RIGHT_ORI_SLICE] = compute_relative_rotation_rot6d(
                abs_chunk[t, RIGHT_ORI_SLICE], abs_chunk[t - 1, RIGHT_ORI_SLICE]
            )

    # delta[0]: pos=0, rotation=identity, gripper=0 (will be overwritten by binary)
    delta[0, LEFT_ORI_SLICE] = identity_rotation_rot6d()
    delta[0, RIGHT_ORI_SLICE] = identity_rotation_rot6d()

    return delta


def apply_gripper_binary(delta: np.ndarray, abs_chunk: np.ndarray,
                          threshold_mm: float = 100.0) -> np.ndarray:
    out = delta.copy()
    out[:, LEFT_GRIP_IDX]  = (abs_chunk[:, LEFT_GRIP_IDX]  > threshold_mm).astype(np.float32)
    out[:, RIGHT_GRIP_IDX] = (abs_chunk[:, RIGHT_GRIP_IDX] > threshold_mm).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Dataset loading (NO STATE)
# ---------------------------------------------------------------------------
def load_single_frame(cfg, dataset_path: Path, episode_idx: int, frame_idx: int):
    data_cfg = cfg.datasets.vla_data
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    _, _, robot_type = mixture_spec[0]

    data_root_dir = dataset_path.parent
    data_name     = dataset_path.name
    data_cfg.data_root_dir = str(data_root_dir)
    logger.info(f"dataset_path={dataset_path}  robot_type={robot_type}")

    dataset = make_LeRobotSingleDataset(
        data_root_dir=data_root_dir,
        data_name=data_name,
        robot_type=robot_type,
        delete_pause_frame=data_cfg.get("delete_pause_frame", False),
        data_cfg=data_cfg,
    )

    traj_ids = dataset.trajectory_ids.tolist()
    if episode_idx not in traj_ids:
        raise ValueError(
            f"episode_idx={episode_idx} not found. Available: "
            f"{traj_ids[:10]}{'...' if len(traj_ids) > 10 else ''} (n={len(traj_ids)})"
        )

    indices_for_ep = [i for i, (tid, _) in enumerate(dataset.all_steps) if tid == episode_idx]
    if not indices_for_ep:
        raise ValueError(f"episode_idx={episode_idx} has no usable frames.")
    if frame_idx < 0 or frame_idx >= len(indices_for_ep):
        raise ValueError(f"frame_idx={frame_idx} out of range [0, {len(indices_for_ep)})")

    flat_index = indices_for_ep[frame_idx]
    _tid, base_index = dataset.all_steps[flat_index]
    logger.info(
        f"Episode {episode_idx}: {len(indices_for_ep)} frames; "
        f"loading frame {frame_idx} (base_index={base_index}, flat_index={flat_index})"
    )

    sample = dataset[flat_index]

    raw_data = dataset.get_step_data(_tid, base_index)
    action_keys_ordered = [
        "action.left_pos", "action.left_ori_6d", "action.left_gripper",
        "action.right_pos", "action.right_ori_6d", "action.right_gripper",
    ]
    raw_parts = []
    for k in action_keys_ordered:
        v = np.asarray(raw_data[k])
        if v.ndim == 1:
            v = v[:, None]
        raw_parts.append(v)
    raw_abs_action = np.concatenate(raw_parts, axis=1).astype(np.float32)
    logger.info(f"[RAW] abs action shape={raw_abs_action.shape}  "
                f"L_pos range=({raw_abs_action[:,0:3].min():.5f}, {raw_abs_action[:,0:3].max():.5f})")

    return sample, raw_abs_action


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(cfg, checkpoint_path: str, device: torch.device) -> Qwen_GR00T:
    logger.info("Building Qwen_GR00T from config")
    model = Qwen_GR00T(cfg)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and "state_dict" not in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _maybe_cumsum(actions: np.ndarray) -> np.ndarray:
    out = actions.copy().astype(np.float32)
    out[:, LEFT_POS_SLICE]  = np.cumsum(actions[:, LEFT_POS_SLICE],  axis=0)
    out[:, LEFT_ORI_SLICE]  = np.cumsum(actions[:, LEFT_ORI_SLICE],  axis=0)
    out[:, RIGHT_POS_SLICE] = np.cumsum(actions[:, RIGHT_POS_SLICE], axis=0)
    out[:, RIGHT_ORI_SLICE] = np.cumsum(actions[:, RIGHT_ORI_SLICE], axis=0)
    return out


def plot_3d_trajectory_dual(gt, pred, save_path, title):
    fig = plt.figure(figsize=(16, 7))
    gt_traj   = _maybe_cumsum(gt)
    pred_traj = _maybe_cumsum(pred)
    for arm_idx, (arm_name, sl) in enumerate(
        [("Left Arm", LEFT_POS_SLICE), ("Right Arm", RIGHT_POS_SLICE)]
    ):
        gt_xyz   = gt_traj[:, sl]
        pred_xyz = pred_traj[:, sl]
        ax = fig.add_subplot(1, 2, arm_idx + 1, projection="3d")
        ax.plot(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],   label="GT",   linewidth=2)
        ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Pred", linewidth=2)
        ax.scatter(*gt_xyz[0],   s=50, marker="o")
        ax.scatter(*pred_xyz[0], s=50, marker="^")
        ax.scatter(*gt_xyz[-1],  s=70, marker="x")
        ax.scatter(*pred_xyz[-1], s=70, marker="*")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(arm_name); ax.legend()
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close(fig)


def _plot_per_arm_curves(gt_arr, pred_arr, save_path, title, ylabel_suffix=None):
    T = min(len(gt_arr), len(pred_arr))
    t = np.arange(T)
    for arm_name, dims in ARM_SPECS:
        fig, axes = plt.subplots(len(dims), 1, figsize=(14, 18), sharex=True)
        for ax_idx, dim in enumerate(dims):
            axes[ax_idx].plot(t, gt_arr[:T, dim],   label="GT",   linewidth=2)
            axes[ax_idx].plot(t, pred_arr[:T, dim], label="Pred", linewidth=2)
            name = ACTION_NAMES[dim]
            if ylabel_suffix and dim in ylabel_suffix:
                name = name + ylabel_suffix[dim]
            axes[ax_idx].set_ylabel(name)
            axes[ax_idx].grid(True, alpha=0.3)
            if ax_idx == 0:
                axes[ax_idx].legend()
        axes[-1].set_xlabel("step")
        fig.suptitle(f"{title} | {arm_name} arm")
        plt.tight_layout()
        out_p = save_path.parent / f"{save_path.stem}_{arm_name}{save_path.suffix}"
        plt.savefig(out_p, dpi=200)
        plt.close(fig)


def plot_raw_delta_curves(gt, pred, save_path, title):
    _plot_per_arm_curves(gt, pred, save_path, title)


def plot_cumulative_curves(gt, pred, save_path, title):
    T = min(len(gt), len(pred))
    gt_cum   = _maybe_cumsum(gt[:T])
    pred_cum = _maybe_cumsum(pred[:T])
    gt_cum[:, LEFT_GRIP_IDX]    = gt[:T,   LEFT_GRIP_IDX]
    gt_cum[:, RIGHT_GRIP_IDX]   = gt[:T,   RIGHT_GRIP_IDX]
    pred_cum[:, LEFT_GRIP_IDX]  = pred[:T, LEFT_GRIP_IDX]
    pred_cum[:, RIGHT_GRIP_IDX] = pred[:T, RIGHT_GRIP_IDX]
    suffix_map = {i: "_cum" for i in range(ACTION_DIM)}
    suffix_map[LEFT_GRIP_IDX]  = "_raw"
    suffix_map[RIGHT_GRIP_IDX] = "_raw"
    _plot_per_arm_curves(gt_cum, pred_cum, save_path, title, ylabel_suffix=suffix_map)


# ---------------------------------------------------------------------------
# Trajectory Error Report (cm)
# ---------------------------------------------------------------------------
def compute_trajectory_error_report(gt_delta: np.ndarray, pred_delta: np.ndarray,
                                     save_path: Path, unit_scale: float = 1.0):
    """Compute positional error between GT and pred trajectories in cm.

    Both gt_delta and pred_delta are [T, 20] in physical delta space.
    We accumulate deltas from step 0 (base frame) to reconstruct trajectories,
    then compute Euclidean distance at steps 10, 20, 30.

    Args:
        unit_scale: multiplier to convert pos units to cm.
                    If pos is already in cm, use 1.0.
                    If pos is in meters, use 100.0.
    """
    T = min(len(gt_delta), len(pred_delta))
    gt_delta = gt_delta[:T]
    pred_delta = pred_delta[:T]

    # Accumulate delta → trajectory (cumsum)
    gt_traj_left   = np.cumsum(gt_delta[:, LEFT_POS_SLICE],  axis=0) * unit_scale
    gt_traj_right  = np.cumsum(gt_delta[:, RIGHT_POS_SLICE], axis=0) * unit_scale
    pred_traj_left  = np.cumsum(pred_delta[:, LEFT_POS_SLICE],  axis=0) * unit_scale
    pred_traj_right = np.cumsum(pred_delta[:, RIGHT_POS_SLICE], axis=0) * unit_scale

    eval_steps = [10, 20, 30]
    results = []

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Trajectory Error Report (position, cm)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Unit scale: pos_unit * {unit_scale} = cm\n")
        f.write(f"Method: cumsum delta from frame 0 as base\n")
        f.write(f"Chunk size: {T} steps\n\n")
        f.write(f"{'Step':>6} | {'Left Err (cm)':>14} | {'Right Err (cm)':>14} | {'Avg (cm)':>10}\n")
        f.write("-" * 60 + "\n")

        for step in eval_steps:
            if step > T:
                f.write(f"{step:>6} | {'N/A (out of range)':>14} |\n")
                continue
            # Use step-1 as index (step 10 means after 10 accumulated deltas)
            idx = step - 1

            err_left = np.linalg.norm(gt_traj_left[idx] - pred_traj_left[idx])
            err_right = np.linalg.norm(gt_traj_right[idx] - pred_traj_right[idx])
            err_avg = (err_left + err_right) / 2.0

            f.write(f"{step:>6} | {err_left:>14.4f} | {err_right:>14.4f} | {err_avg:>10.4f}\n")
            results.append({
                "step": step,
                "left_err_cm": float(err_left),
                "right_err_cm": float(err_right),
                "avg_err_cm": float(err_avg),
            })

        f.write("-" * 60 + "\n\n")

        # Per-step breakdown for left arm
        f.write("Left arm trajectory (cumsum xyz, cm):\n")
        f.write(f"  {'Step':>4} | {'GT_x':>8} {'GT_y':>8} {'GT_z':>8} | {'Pred_x':>8} {'Pred_y':>8} {'Pred_z':>8} | {'Err':>8}\n")
        for step in eval_steps:
            if step > T:
                break
            idx = step - 1
            gt_p = gt_traj_left[idx]
            pr_p = pred_traj_left[idx]
            err = np.linalg.norm(gt_p - pr_p)
            f.write(f"  {step:>4} | {gt_p[0]:>8.4f} {gt_p[1]:>8.4f} {gt_p[2]:>8.4f} | "
                    f"{pr_p[0]:>8.4f} {pr_p[1]:>8.4f} {pr_p[2]:>8.4f} | {err:>8.4f}\n")

        f.write("\nRight arm trajectory (cumsum xyz, cm):\n")
        f.write(f"  {'Step':>4} | {'GT_x':>8} {'GT_y':>8} {'GT_z':>8} | {'Pred_x':>8} {'Pred_y':>8} {'Pred_z':>8} | {'Err':>8}\n")
        for step in eval_steps:
            if step > T:
                break
            idx = step - 1
            gt_p = gt_traj_right[idx]
            pr_p = pred_traj_right[idx]
            err = np.linalg.norm(gt_p - pr_p)
            f.write(f"  {step:>4} | {gt_p[0]:>8.4f} {gt_p[1]:>8.4f} {gt_p[2]:>8.4f} | "
                    f"{pr_p[0]:>8.4f} {pr_p[1]:>8.4f} {pr_p[2]:>8.4f} | {err:>8.4f}\n")

    # Also save as JSON for programmatic use
    json_path = save_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"unit": "cm", "unit_scale": unit_scale, "errors": results}, f, indent=2)

    logger.info(f"Trajectory error report saved to: {save_path}")
    return results


# ---------------------------------------------------------------------------
# Text report (per-dim metrics)
# ---------------------------------------------------------------------------
def save_text_report(gt, pred, save_path, space_note="unnormalized delta"):
    T = min(len(gt), len(pred))
    gt, pred = gt[:T], pred[:T]
    mae = np.mean(np.abs(gt - pred), axis=0)
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=0))
    corr, sign_match = [], []
    for i in range(ACTION_DIM):
        g, p = gt[:, i], pred[:, i]
        if np.std(g) < 1e-8 or np.std(p) < 1e-8:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(g, p)[0, 1]))
        sign_match.append(float(np.mean(np.sign(g) == np.sign(p))))

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Offline Eval Report — Galbot Bimanual NO STATE (delta model)\n")
        f.write("=" * 70 + "\n")
        f.write(f"comparison_space = {space_note}\n")
        f.write(f"T = {T}\n\n")
        f.write("Per-dim metrics:\n" + "-" * 70 + "\n")
        for i in range(ACTION_DIM):
            f.write(
                f"{ACTION_NAMES[i]:14s} | MAE={mae[i]:.6f} | RMSE={rmse[i]:.6f} "
                f"| Corr={corr[i]:.6f} | SignMatch={sign_match[i]:.4f}\n"
            )
        f.write("\nEnd displacement (cumsum of delta, xyz only):\n")
        gt_traj   = _maybe_cumsum(gt)
        pred_traj = _maybe_cumsum(pred)
        for arm_name, sl in [("Left", LEFT_POS_SLICE), ("Right", RIGHT_POS_SLICE)]:
            f.write(f"{arm_name} GT   end xyz = {gt_traj[-1, sl].tolist()}\n")
            f.write(f"{arm_name} Pred end xyz = {pred_traj[-1, sl].tolist()}\n")


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------
def _save_pil_or_array(img, save_path: Path):
    if isinstance(img, Image.Image):
        img.save(save_path); return
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Offline eval for QwenGR00T galbot bimanual NO-STATE delta-mode policy."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--episode-idx", type=int, required=True)
    parser.add_argument("--frame-idx",   type=int, required=True)
    parser.add_argument("--config-yaml", type=str, default=str(DEFAULT_CONFIG_YAML))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--unit-scale", type=float, default=1.0,
                        help="Multiplier to convert pos units to cm (1.0 if already cm, 100.0 if meters)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    OmegaConf.set_struct(cfg, False)

    ckpt_path = Path(args.checkpoint).resolve()
    out_root  = Path(args.out_dir).resolve() if args.out_dir else ckpt_path.with_suffix("")
    out_dir   = out_root / f"ep{args.episode_idx:04d}_frame{args.frame_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    dataset_path = Path(args.dataset_path).resolve()

    # 1. Load sample + raw abs action chunk
    sample, raw_abs_action = load_single_frame(
        cfg, dataset_path, args.episode_idx, args.frame_idx
    )

    # 2. Load normalization stats
    stats = _load_stats(dataset_path)
    gripper_th = float(cfg.datasets.vla_data.get("gripper_binary_threshold", 100.0))

    # 3. Save input images
    images = sample["image"]
    image_keys_order = ["left_wrist", "right_wrist"]
    for i, img in enumerate(images):
        name = image_keys_order[i] if i < len(image_keys_order) else f"view{i}"
        _save_pil_or_array(img, out_dir / f"input_{name}.png")
    logger.info(f"Saved {len(images)} input view(s)")

    # 4. Save prompt
    prompt = sample.get("lang", "")
    (out_dir / "prompt.txt").write_text(str(prompt), encoding="utf-8")
    logger.info(f"Prompt: {prompt}")

    # 5. GT = abs→delta, binary gripper
    gt_delta = abs_to_delta(raw_abs_action)
    gt_delta = apply_gripper_binary(gt_delta, raw_abs_action, gripper_th)
    logger.info(
        f"[GT] delta L_pos range=({gt_delta[:,0:3].min():.5f},{gt_delta[:,0:3].max():.5f}) "
        f"R_pos=({gt_delta[:,10:13].min():.5f},{gt_delta[:,10:13].max():.5f})"
    )

    # 6. Inference (NO STATE — model handles state=None automatically)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, str(ckpt_path), device)
    predict_output = model.predict_action(examples=[sample])
    normed_pred = np.asarray(predict_output["normalized_actions"][0], dtype=np.float32)
    if normed_pred.ndim != 2 or normed_pred.shape[-1] != ACTION_DIM:
        raise ValueError(f"Pred shape {normed_pred.shape} != [T, {ACTION_DIM}]")
    logger.info(f"Pred (normalized) shape: {normed_pred.shape}")

    # 7. Denormalize prediction
    pred_delta = normalized_to_delta(normed_pred, stats)
    logger.info(
        f"[Pred] delta L_pos range=({pred_delta[:,0:3].min():.5f},{pred_delta[:,0:3].max():.5f}) "
        f"R_pos=({pred_delta[:,10:13].min():.5f},{pred_delta[:,10:13].max():.5f})"
    )

    # 8. Align lengths
    T = min(len(gt_delta), len(pred_delta))
    gt_delta   = gt_delta[:T]
    pred_delta = pred_delta[:T]
    logger.info(f"Comparing T={T} steps in unnormalized delta space")

    # 9. Save arrays
    np.save(out_dir / "gt_actions.npy",   gt_delta)
    np.save(out_dir / "pred_actions.npy", pred_delta)

    # 10. Plots
    tag = f"ep={args.episode_idx}, frame={args.frame_idx}, T={T} [unnorm delta, NO STATE]"
    plot_3d_trajectory_dual(
        gt_delta, pred_delta, out_dir / "traj3d_xyz.png",
        f"3D xyz cumsum (dual arm, unnorm delta) | {tag}"
    )
    plot_raw_delta_curves(
        gt_delta, pred_delta, out_dir / "raw_action_curves.png",
        f"Raw delta actions (unnormalized) | {tag}"
    )
    plot_cumulative_curves(
        gt_delta, pred_delta, out_dir / "cumulative_action_curves.png",
        f"Cumulative delta (unnormalized) | {tag}"
    )

    # 11. Per-dim report
    save_text_report(gt_delta, pred_delta, out_dir / "report.txt",
                     space_note="unnormalized delta (physical units, NO STATE)")

    # 12. Trajectory error report (cm) at steps 10, 20, 30
    errors = compute_trajectory_error_report(
        gt_delta, pred_delta, out_dir / "trajectory_error.txt",
        unit_scale=args.unit_scale,
    )
    for e in errors:
        logger.info(f"  Step {e['step']:>2}: L={e['left_err_cm']:.4f}cm  "
                    f"R={e['right_err_cm']:.4f}cm  Avg={e['avg_err_cm']:.4f}cm")

    logger.info(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
