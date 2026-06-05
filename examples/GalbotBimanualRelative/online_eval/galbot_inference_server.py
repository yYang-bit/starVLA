#!/usr/bin/env python3
"""
StarVLA QwenGR00T Dual-Arm Inference Server
============================================

Server 端职责:
  1. 接收 client 发来的双臂图像 + abs eef pose
  2. 计算 dual_relative_pose (左臂在右臂坐标系下的相对位姿)
  3. 模型推理得到 normalized delta_action
  4. 反归一化后下发物理单位的 delta_action

Client 端建议:
  - 在 client 端将图像 resize 到模型输入尺寸 (e.g. 224x224) 再发送，减少传输量
  - 发送双臂 abs eef pose: left [xyz(3), rot6d(6), gripper(1)] + right [xyz(3), rot6d(6), gripper(1)]

HTTP API:
  POST /infer
  Request JSON:
    {
      "images": {
        "left_wrist": "<base64_encoded_jpeg_or_array>",
        "right_wrist": "<base64_encoded_jpeg_or_array>"
      },
      "abs_eef_pose": {
        "left": [x, y, z, r00, r01, r02, r10, r11, r12, gripper],   # 10D
        "right": [x, y, z, r00, r01, r02, r10, r11, r12, gripper]   # 10D
      }
    }
  Response JSON:
    {
      "delta_action": [20D list],  # 物理单位的 delta action
      "inference_time_ms": float
    }

Usage:
  python galbot_inference_server.py \\
      --checkpoint /path/to/steps_80000_pytorch_model.pt \\
      --dataset_path /path/to/lerobot_dataset \\
      --port 5000 \\
      --host 0.0.0.0
"""

# ==========================================
# Configuration (modify these paths)
# ==========================================
DEFAULT_CHECKPOINT = "/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action/galbot_bimanual_self_0529/steps_80000_pytorch_model.pt"
DEFAULT_DATASET_PATH = "/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0529_359piece"
DEFAULT_STATS_FILENAME = "stats.json"  # stats.json or stats_delta_chunk30.json
DEFAULT_PORT = 5000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_INCLUDE_STATE = True  # True for Relative mode, False for NoState mode
# ==========================================

import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from omegaconf import OmegaConf

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ==========================================
# Global Context
# ==========================================
class GlobalContext:
    model = None
    stats = None
    device = None
    include_state = True  # True for Relative, False for NoState
    action_horizon = 30


ctx = GlobalContext()


# ==========================================
# Image Decoding Utilities
# ==========================================
def decode_image(value: Any) -> np.ndarray:
    """Decode base64 or array to numpy RGB image [H, W, 3]."""
    if isinstance(value, str):
        # Base64 string
        if value.startswith("data:image/"):
            value = value.split(",", 1)[1]
        img_bytes = base64.b64decode(value)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)
    else:
        # Array
        arr = np.asarray(value, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr
        elif arr.ndim == 3 and arr.shape[0] == 3:
            return arr.transpose(1, 2, 0)
        else:
            raise ValueError(f"Invalid image array shape: {arr.shape}")


# ==========================================
# Relative Pose Computation
# ==========================================
def compute_dual_relative_pose(left_pose: np.ndarray, right_pose: np.ndarray) -> np.ndarray:
    """
    Compute left arm's pose in right arm's coordinate frame.

    Args:
        left_pose: [10] = [xyz(3), rot6d(6), gripper(1)]
        right_pose: [10] = [xyz(3), rot6d(6), gripper(1)]

    Returns:
        dual_relative_pose: [9] = [rel_xyz(3), rel_rot6d(6)]
    """
    from starVLA.dataloader.gr00t_lerobot.transform.rotation_utils import (
        rot6d_to_matrix,
        matrix_to_rot6d,
    )

    left_xyz = left_pose[:3]
    left_rot6d = left_pose[3:9]
    right_xyz = right_pose[:3]
    right_rot6d = right_pose[3:9]

    # Relative position: R_right^T @ (p_left - p_right)
    R_right = rot6d_to_matrix(right_rot6d[None, :])[0]  # (3, 3)
    rel_xyz = R_right.T @ (left_xyz - right_xyz)

    # Relative rotation: R_right^T @ R_left
    R_left = rot6d_to_matrix(left_rot6d[None, :])[0]  # (3, 3)
    R_rel = R_right.T @ R_left
    rel_rot6d = matrix_to_rot6d(R_rel[None, :])[0]  # (6,)

    return np.concatenate([rel_xyz, rel_rot6d]).astype(np.float32)


# ==========================================
# Unnormalization
# ==========================================
def unnormalize_delta_action(normalized_action: np.ndarray, stats: dict) -> np.ndarray:
    """
    Inverse q99 normalization for pos dimensions only.

    Args:
        normalized_action: [T, 20] normalized delta action from model
        stats: dict with keys "action" -> {"q01": list, "q99": list}

    Returns:
        physical_action: [T, 20] in physical units (mm for pos, ori_6d unchanged, gripper binary)
    """
    action_stats = stats["statistics"]["action"]
    q01 = np.array(action_stats["q01"], dtype=np.float32)
    q99 = np.array(action_stats["q99"], dtype=np.float32)

    # Only unnormalize pos dimensions: [0:3] left_pos, [10:13] right_pos
    # ori_6d and gripper are not normalized
    pos_indices = list(range(0, 3)) + list(range(10, 13))

    physical = normalized_action.copy()
    for idx in pos_indices:
        # Inverse: physical = normalized * (q99 - q01) / 2
        physical[:, idx] = normalized_action[:, idx] * (q99[idx] - q01[idx]) / 2.0

    return physical


# ==========================================
# Model Inference
# ==========================================
@torch.no_grad()
def run_inference(images: dict[str, np.ndarray], abs_eef_pose: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run model inference.

    Args:
        images: {"left_wrist": [H,W,3], "right_wrist": [H,W,3]} uint8 RGB
        abs_eef_pose: {"left": [10], "right": [10]} float32

    Returns:
        delta_action: [action_horizon, 20] physical units
    """
    # 1. Prepare images (to tensor, normalize to [0, 1])
    left_img = torch.from_numpy(images["left_wrist"]).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    right_img = torch.from_numpy(images["right_wrist"]).permute(2, 0, 1).float() / 255.0  # (3, H, W)

    # Add batch + time dim: (1, 1, 3, H, W)
    left_img = left_img.unsqueeze(0).unsqueeze(0).to(ctx.device)
    right_img = right_img.unsqueeze(0).unsqueeze(0).to(ctx.device)

    # 2. Prepare state (if Relative mode)
    if ctx.include_state:
        dual_rel_pose = compute_dual_relative_pose(abs_eef_pose["left"], abs_eef_pose["right"])
        state = torch.from_numpy(dual_rel_pose).unsqueeze(0).unsqueeze(0).to(ctx.device)  # (1, 1, 9)
    else:
        state = torch.zeros((1, 1, 0), device=ctx.device)  # NoState mode

    # 3. Prepare task instruction (empty for now)
    task_instruction = [""]

    # 4. Model forward
    normalized_actions = ctx.model(
        rgb={"observation.images.left_wrist": left_img, "observation.images.right_wrist": right_img},
        state=state,
        task_instructions=task_instruction,
    )  # (1, action_horizon, 20)

    normalized_actions = normalized_actions.squeeze(0).cpu().numpy()  # (action_horizon, 20)

    # 5. Unnormalize
    physical_actions = unnormalize_delta_action(normalized_actions, ctx.stats)

    return physical_actions


# ==========================================
# HTTP Endpoints
# ==========================================
@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json()

        # Parse images
        images = {
            "left_wrist": decode_image(data["images"]["left_wrist"]),
            "right_wrist": decode_image(data["images"]["right_wrist"]),
        }

        # Parse abs eef pose
        abs_eef_pose = {
            "left": np.array(data["abs_eef_pose"]["left"], dtype=np.float32),
            "right": np.array(data["abs_eef_pose"]["right"], dtype=np.float32),
        }

        # Run inference
        t0 = time.time()
        delta_action = run_inference(images, abs_eef_pose)
        inference_time_ms = (time.time() - t0) * 1000

        return jsonify({
            "delta_action": delta_action.tolist(),
            "inference_time_ms": inference_time_ms,
        })

    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": ctx.model is not None})


# ==========================================
# Initialization
# ==========================================
def load_model(checkpoint_path: str, dataset_path: str, stats_filename: str, include_state: bool):
    """Load model and stats."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Load config
    if "model_config" in ckpt:
        config = ckpt["model_config"]
    else:
        raise ValueError("Checkpoint missing 'model_config'")

    config = OmegaConf.create(config)

    # Override include_state
    config.include_state = include_state

    # Initialize model
    model = Qwen_GR00T(config)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(ctx.device)
    logger.info(f"Model loaded on {ctx.device}")

    # Load stats
    stats_path = Path(dataset_path) / "meta" / stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)
    logger.info(f"Loaded stats from: {stats_path}")

    ctx.model = model
    ctx.stats = stats
    ctx.include_state = include_state
    ctx.action_horizon = config.chunk_size


def main():
    parser = argparse.ArgumentParser(description="StarVLA Dual-Arm Inference Server")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to lerobot dataset (for stats.json)")
    parser.add_argument("--stats_filename", type=str, default=DEFAULT_STATS_FILENAME, help="Stats filename in <dataset_path>/meta/")
    parser.add_argument("--include_state", action="store_true", default=DEFAULT_INCLUDE_STATE, help="Use Relative mode (with state input)")
    parser.add_argument("--no_state", dest="include_state", action="store_false", help="Use NoState mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    args = parser.parse_args()

    load_model(args.checkpoint, args.dataset_path, args.stats_filename, args.include_state)

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Mode: {'Relative (with state)' if args.include_state else 'NoState (vision only)'}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
