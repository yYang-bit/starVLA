#!/usr/bin/env python3
"""
离线计算 GalbotBimanualSelf 数据集 stats.json。

支持三种 self_mode，模拟训练时的 chunk 采样 + action 变换逻辑，
保证统计值与训练时实际输入数据严格对齐。

Action transform:
    - pos: 算术差分
    - rotation_6d: SO(3) 相对旋转 (R[t] @ R[base]^T)
    - gripper: 二值化（在 abs 值上应用 threshold）

用法:
    python compute_galbot_stats_self_mode.py \
        --dataset_dir /path/to/lerobot_dataset \
        --self_mode delta \
        --chunk_size 16 \
        --gripper_threshold 100.0 \
        [--output_path /path/to/stats.json]

    # 输出默认写到 <dataset_dir>/meta/stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add rotation utils to path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.transform.rotation_utils import (
    compute_relative_rotation_rot6d,
    identity_rotation_rot6d,
)

COL_ACTION = "action"
COL_OBS_STATE = "observation.state"
COL_DUAL_REL = "observation.dual_relative_pose"

ACTION_DIM = 20
STATE_DIM = 9
GRIPPER_DIMS = [9, 19]

# Rotation_6d column indices (left: 3-9, right: 13-19)
LEFT_ORI_SLICE = slice(3, 9)
RIGHT_ORI_SLICE = slice(13, 19)
# Position column indices
LEFT_POS_SLICE = slice(0, 3)
RIGHT_POS_SLICE = slice(10, 13)


def load_all_actions(dataset_dir: Path) -> np.ndarray:
    """加载所有 parquet 的 abs action，拼接为 (N, 20)。"""
    parquet_paths = sorted(dataset_dir.rglob("data/**/*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"未找到 parquet 文件: {dataset_dir}/data/")

    chunks = []
    for p in tqdm(parquet_paths, desc="加载 parquet"):
        df = pd.read_parquet(p, columns=[COL_ACTION])
        chunks.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df[COL_ACTION]]))
    return np.concatenate(chunks, axis=0)


def _batch_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Vectorized rot6d → rotation matrix. Input: (N, 6), Output: (N, 3, 3)."""
    rot6d = rot6d.astype(np.float64)
    x_raw = rot6d[:, :3]
    y_raw = rot6d[:, 3:]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    y_orth = y_raw - np.sum(y_raw * x, axis=-1, keepdims=True) * x
    y = y_orth / (np.linalg.norm(y_orth, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y, axis=-1)
    R = np.stack([x, y, z], axis=-1)  # (N, 3, 3)
    return R


def _batch_matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation matrix → rot6d. Input: (N, 3, 3), Output: (N, 6)."""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1).astype(np.float32)


def _batch_relative_rotation(rot6d_t: np.ndarray, rot6d_base: np.ndarray) -> np.ndarray:
    """Vectorized SO(3) relative rotation. Input: (N, 6) each, Output: (N, 6).
    R_rel = R_t @ R_base^T
    """
    R_t = _batch_rot6d_to_matrix(rot6d_t)        # (N, 3, 3)
    R_base = _batch_rot6d_to_matrix(rot6d_base)  # (N, 3, 3)
    R_base_T = R_base.transpose(0, 2, 1)         # (N, 3, 3)
    R_rel = R_t @ R_base_T                       # (N, 3, 3)
    return _batch_matrix_to_rot6d(R_rel)


def apply_self_mode_action(
    abs_action: np.ndarray,
    self_mode: str,
    chunk_size: int,
    gripper_threshold: float,
) -> np.ndarray:
    """
    向量化版本：模拟训练时的 chunk 采样 + action 变换 + gripper 二值化。

    策略：先构建所有 chunk 的 (N, chunk_size, 20) 张量，再批量变换。
    """
    total = abs_action.shape[0]
    print(f"    构建 chunk 张量 ({total}, {chunk_size}, 20) ...")

    # 构建 padded action（尾部 pad 一个 chunk_size 的重复帧）
    padded = np.concatenate([abs_action, np.repeat(abs_action[-1:], chunk_size, axis=0)], axis=0)

    # 构建所有 chunk 的索引: (N, chunk_size)
    indices = np.arange(chunk_size)[None, :] + np.arange(total)[:, None]  # (N, chunk_size)
    chunks = padded[indices]  # (N, chunk_size, 20)

    # gripper 二值化（在 abs 值上）
    for idx in GRIPPER_DIMS:
        chunks[:, :, idx] = (chunks[:, :, idx] > gripper_threshold).astype(np.float32)

    N = chunks.shape[0]

    if self_mode == "chunk_relative":
        print(f"    chunk_relative: pos 差分 ...")
        # pos: chunks[:, t, :] -= chunks[:, 0, :]
        chunks[:, :, LEFT_POS_SLICE] -= chunks[:, 0:1, LEFT_POS_SLICE]
        chunks[:, :, RIGHT_POS_SLICE] -= chunks[:, 0:1, RIGHT_POS_SLICE]

        # rotation_6d: R[t] @ R[0]^T (vectorized over N*chunk_size)
        print(f"    chunk_relative: SO(3) 旋转 (left) ...")
        for t_idx in tqdm(range(1, chunk_size), desc="    left_ori chunk_rel"):
            chunks[:, t_idx, LEFT_ORI_SLICE] = _batch_relative_rotation(
                chunks[:, t_idx, LEFT_ORI_SLICE],
                chunks[:, 0, LEFT_ORI_SLICE],
            )
        print(f"    chunk_relative: SO(3) 旋转 (right) ...")
        for t_idx in tqdm(range(1, chunk_size), desc="    right_ori chunk_rel"):
            chunks[:, t_idx, RIGHT_ORI_SLICE] = _batch_relative_rotation(
                chunks[:, t_idx, RIGHT_ORI_SLICE],
                chunks[:, 0, RIGHT_ORI_SLICE],
            )
        # t=0: R[0] @ R[0]^T = I
        chunks[:, 0, LEFT_ORI_SLICE] = identity_rotation_rot6d()
        chunks[:, 0, RIGHT_ORI_SLICE] = identity_rotation_rot6d()

    elif self_mode == "delta":
        print(f"    delta: pos 差分 ...")
        # pos: delta[t] = pos[t] - pos[t-1], delta[0] = 0
        pos_left = chunks[:, :, LEFT_POS_SLICE].copy()
        chunks[:, 1:, LEFT_POS_SLICE] = pos_left[:, 1:, :] - pos_left[:, :-1, :]
        chunks[:, 0, LEFT_POS_SLICE] = 0

        pos_right = chunks[:, :, RIGHT_POS_SLICE].copy()
        chunks[:, 1:, RIGHT_POS_SLICE] = pos_right[:, 1:, :] - pos_right[:, :-1, :]
        chunks[:, 0, RIGHT_POS_SLICE] = 0

        # rotation_6d: R[t] @ R[t-1]^T, R[0] = I
        # 需要保留原始 rot6d 用于计算（从后往前会覆盖，所以先 copy）
        ori_left_orig = chunks[:, :, LEFT_ORI_SLICE].copy()
        ori_right_orig = chunks[:, :, RIGHT_ORI_SLICE].copy()

        print(f"    delta: SO(3) 旋转 (left + right, {chunk_size-1} steps) ...")
        for t_idx in tqdm(range(1, chunk_size), desc="    ori delta"):
            chunks[:, t_idx, LEFT_ORI_SLICE] = _batch_relative_rotation(
                ori_left_orig[:, t_idx, :],
                ori_left_orig[:, t_idx - 1, :],
            )
            chunks[:, t_idx, RIGHT_ORI_SLICE] = _batch_relative_rotation(
                ori_right_orig[:, t_idx, :],
                ori_right_orig[:, t_idx - 1, :],
            )

        chunks[:, 0, LEFT_ORI_SLICE] = identity_rotation_rot6d()
        chunks[:, 0, RIGHT_ORI_SLICE] = identity_rotation_rot6d()

    # 展平为 (N * chunk_size, 20)
    print(f"    展平 → ({N * chunk_size}, 20)")
    return chunks.reshape(-1, 20)


def compute_stats(data: np.ndarray) -> dict:
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std":  np.std(data,  axis=0).tolist(),
        "min":  np.min(data,  axis=0).tolist(),
        "max":  np.max(data,  axis=0).tolist(),
        "q01":  np.quantile(data, 0.01, axis=0).tolist(),
        "q99":  np.quantile(data, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="离线计算 self-mode stats.json")
    parser.add_argument("--dataset_dir", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--output_path", default=None, help="输出路径，默认 <dataset_dir>/meta/stats.json")
    parser.add_argument("--self_mode", default="abs", choices=["abs", "delta", "chunk_relative"])
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--gripper_threshold", type=float, default=100.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_path) if args.output_path else dataset_dir / "meta" / "stats.json"

    # abs action 统计（state 列不参与 transform，直接算原始 stats）
    print(f"加载数据: {dataset_dir}")
    abs_action = load_all_actions(dataset_dir)
    print(f"  abs_action: {abs_action.shape}")

    # 对 action 应用 self-mode 变换 + gripper 二值化
    if args.self_mode == "abs":
        # abs 模式只有 gripper 二值化
        action = abs_action.copy()
        for idx in GRIPPER_DIMS:
            action[:, idx] = (action[:, idx] > args.gripper_threshold).astype(np.float32)
    else:
        print(f"  应用 self_mode={args.self_mode}, chunk_size={args.chunk_size} ...")
        action = apply_self_mode_action(abs_action, args.self_mode, args.chunk_size, args.gripper_threshold)
    print(f"  action after transform: {action.shape}")

    # dual_relative_pose + observation.state 统计（不受 action_mode 影响，直接算原始值）
    state_paths = sorted(dataset_dir.rglob("data/**/*.parquet"))
    dual_rel_chunks = []
    obs_state_chunks = []
    for p in tqdm(state_paths, desc="加载 state"):
        df = pd.read_parquet(p, columns=[COL_DUAL_REL, COL_OBS_STATE])
        dual_rel_chunks.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df[COL_DUAL_REL]]))
        obs_state_chunks.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df[COL_OBS_STATE]]))
    state_data = np.concatenate(dual_rel_chunks, axis=0)
    obs_state_data = np.concatenate(obs_state_chunks, axis=0)
    print(f"  dual_relative_pose: {state_data.shape}")
    print(f"  observation.state: {obs_state_data.shape}")

    # 写入 stats.json
    stats = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},  # 框架用 data_cfg.action_mode（默认"abs"）做缓存校验，与 self_mode 解耦
        "statistics": {
            COL_ACTION:     compute_stats(action),
            COL_DUAL_REL:   compute_stats(state_data),
            COL_OBS_STATE:  compute_stats(obs_state_data),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nstats.json 已写入: {output_path}")

    # 摘要
    for col, label in [(COL_ACTION, "action"), (COL_DUAL_REL, "state")]:
        s = stats["statistics"][col]
        print(f"\n  [{label}]")
        print(f"    q01: {[round(v, 4) for v in s['q01']]}")
        print(f"    q99: {[round(v, 4) for v in s['q99']]}")

    print(f"\n  ✅ 请在 yaml 中设置:")
    print(f"     datasets.vla_data.self_mode: {args.self_mode}")


if __name__ == "__main__":
    main()