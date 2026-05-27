#!/usr/bin/env python3
"""
将 LeRobot v3 的 tasks.jsonl 转换为 tasks.parquet（框架要求）。

用法：
    python convert_tasks_jsonl_to_parquet.py \
        --dataset_dir /path/to/lerobot_dataset
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="LeRobot 数据集根目录")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    jsonl_path = dataset_dir / "meta" / "tasks.jsonl"
    parquet_path = dataset_dir / "meta" / "tasks.parquet"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {jsonl_path}")

    # 读取 jsonl
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    # 转为 DataFrame，设置 task_index 为索引
    df = pd.DataFrame(tasks)
    df = df.set_index("task_index")
    print(f"Loaded {len(df)} tasks from {jsonl_path}")
    print(df.head())

    # 写入 parquet
    df.to_parquet(parquet_path, index=True)
    print(f"\n✅ Saved to {parquet_path}")


if __name__ == "__main__":
    main()
