#!/usr/bin/env python3
"""
离线计算 GalbotBimanualSelf 数据集 stats.json。

支持三种 self_mode，模拟训练时的 chunk 采样 + action 变换逻辑，
保证统计值与训练时实际输入数据严格对齐。

用法：
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
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

COL_ACTION = "action"
COL_OBS_STATE = "observation.state"
COL_DUAL_REL = "observation.dual_relative_pose"

ACTION_DIM = 20
STATE_DIM = 9
GRIPPER_DIMS = [9, 19]


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


def apply_self_mode_action(
    abs_action: np.ndarray,
    self_mode: str,
    chunk_size: int,
    gripper_threshold: float,
) -> np.ndarray:
    """
    模拟训练时的 chunk 采样 + action 变换 + gripper 二值化。

    输出形状: (N_chunks * chunk_size, 20)，即所有 chunk 帧展平后的数据。
    """
    total = abs_action.shape[0]
    out_chunks = []

    for t in range(total):
        # 取 chunk: [t, t+chunk_size)
        end = t + chunk_size
        chunk = abs_action[t:end].copy()
        if end > total:
            # 尾部 padding：用最后一帧填充
            pad_len = end - total
            chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="edge")
            # 调整 end 使后面的循环正确
            end = total

        # gripper 二值化必须在 self_mode 变换之前，作用在 abs 值上
        for idx in GRIPPER_DIMS:
            chunk[:, idx] = (chunk[:, idx] > gripper_threshold).astype(np.float32)

        if self_mode == "chunk_relative":
            # pos/ori_6d: action[t] -= action[0]；gripper 已二值化，不参与
            non_gripper = [i for i in range(chunk.shape[1]) if i not in GRIPPER_DIMS]
            chunk[:, non_gripper] = chunk[:, non_gripper] - chunk[0:1, non_gripper]

        elif self_mode == "delta":
            # pos/ori_6d: action[t] = action[t] - action[t-1], action[0] = 0；gripper 不做差分
            non_gripper = [i for i in range(chunk.shape[1]) if i not in GRIPPER_DIMS]
            tmp = chunk[:, non_gripper].astype(np.float64).copy()
            for i in range(chunk_size - 1, 0, -1):
                tmp[i] = tmp[i] - tmp[i - 1]
            tmp[0] = 0
            chunk[:, non_gripper] = tmp.astype(np.float32)

        out_chunks.append(chunk)

    return np.concatenate(out_chunks, axis=0)


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