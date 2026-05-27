#!/usr/bin/env python3
"""
将 LeRobot v2.1 的 episodes.jsonl 转换为 v3 格式的 meta/episodes/chunk-000/episodes.parquet。

v2.1 格式: meta/episodes.jsonl（每行一个 episode JSON）
v3 格式: meta/episodes/chunk-000/episodes.parquet（按 chunk 分组的 parquet 文件）

用法：
    python convert_episodes_jsonl_to_parquet.py \
        --dataset_dir /path/to/lerobot_dataset
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="每个 parquet 文件包含的 episode 数量（默认 1000）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    jsonl_path = dataset_dir / "meta" / "episodes.jsonl"
    episodes_dir = dataset_dir / "meta" / "episodes"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {jsonl_path}")

    # 读取 jsonl
    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    print(f"Loaded {len(episodes)} episodes from {jsonl_path}")

    # 读取 info.json 获取 chunks_size
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", args.chunk_size)
    print(f"Using chunks_size={chunks_size}")

    # 转换为 DataFrame
    # v3 期望的列：episode_index, tasks (list), length, data/chunk_index, data/file_index
    # 以及可选的 videos/{key}/from_timestamp
    rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size
        file_idx = ep_idx % chunks_size  # 文件内序号

        row = {
            "episode_index": ep_idx,
            "tasks": ep.get("tasks", []),
            "length": ep["length"],
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 按 chunk 分组写入
    for chunk_idx in sorted(df["data/chunk_index"].unique()):
        chunk_df = df[df["data/chunk_index"] == chunk_idx].reset_index(drop=True)
        chunk_dir = episodes_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / "episodes.parquet"
        chunk_df.to_parquet(parquet_path, index=False)
        print(f"  {parquet_path} ({len(chunk_df)} episodes)")

    print(f"\n✅ Episodes parquet 已写入: {episodes_dir}")


if __name__ == "__main__":
    main()
