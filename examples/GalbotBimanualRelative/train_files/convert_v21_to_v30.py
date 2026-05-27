#!/usr/bin/env python3
"""
LeRobot v2.1 → v3.0 本地转换脚本（简化版，不做文件合并）。

参考官方脚本：lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
官方脚本会合并多个 episode 到一个大文件；本脚本保留每集一文件，仅做格式转换。

转换内容：
  1. data/chunk-000/episode_000000.parquet → data/chunk-000/file_000000.parquet
  2. videos/chunk-000/CAM/episode_000000.mp4 → videos/CAM/chunk-000/file_000000.mp4
  3. meta/episodes.jsonl → meta/episodes/chunk-000/episodes.parquet
  4. meta/tasks.jsonl → meta/tasks/chunk-000/file_000.parquet
  5. meta/episodes_stats.jsonl → flatten 到 episodes.parquet
  6. meta/info.json 更新 codebase_version / data_path / video_path / features.fps
  7. 删除旧文件

用法：
    python convert_v21_to_v30.py \
        --dataset_dir /path/to/lerobot_dataset

环境：lerobot_v2_zbl 或 lerobot_v3
"""

import argparse
import json
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa

V21 = "v2.1"
V30 = "v3.0"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file_{file_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:06d}.mp4"


def load_jsonlines(fpath: Path) -> list[dict]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def validate_version(root: Path) -> None:
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    version = info.get("codebase_version", "unknown")
    if version != V21:
        raise ValueError(f"数据集 codebase_version={version!r}，期望 {V21!r}")


def get_video_keys(root: Path) -> list[str]:
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    return [k for k, ft in info["features"].items() if ft["dtype"] == "video"]


def convert_data(root: Path) -> list[dict]:
    """重命名 data parquet，返回每集元数据。"""
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    if not ep_paths:
        raise RuntimeError(f"未找到 data parquet: {data_dir}")

    episodes_metadata = []
    num_frames = 0

    for ep_path in ep_paths:
        # 从文件名提取 episode_index
        ep_name = ep_path.stem  # e.g. episode_000000
        ep_idx = int(ep_name.split("_")[-1])
        chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
        file_idx = ep_idx % DEFAULT_CHUNK_SIZE

        # 读取帧数
        df = pd.read_parquet(ep_path)
        ep_num_frames = len(df)

        # 新路径
        new_path = data_dir / f"chunk-{chunk_idx:03d}" / f"file_{file_idx:06d}.parquet"
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # 重命名（如果新旧路径不同）
        if ep_path != new_path:
            ep_path.rename(new_path)

        episodes_metadata.append({
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        })
        num_frames += ep_num_frames

    return episodes_metadata


def convert_videos(root: Path) -> list[dict] | None:
    """移动视频文件到新目录结构。"""
    video_keys = get_video_keys(root)
    if not video_keys:
        return None

    all_eps_metadata = []

    for video_key in sorted(video_keys):
        videos_dir = root / "videos"
        ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))
        if not ep_paths:
            raise RuntimeError(f"未找到视频: {videos_dir}/*/{video_key}/*.mp4")

        eps_metadata = []

        for ep_path in ep_paths:
            ep_name = ep_path.stem  # e.g. episode_000000
            ep_idx = int(ep_name.split("_")[-1])
            chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
            file_idx = ep_idx % DEFAULT_CHUNK_SIZE

            # v3 路径: videos/CAMERA/chunk-000/file_000000.mp4
            new_path = root / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file_{file_idx:06d}.mp4"
            new_path.parent.mkdir(parents=True, exist_ok=True)

            if ep_path != new_path:
                ep_path.rename(new_path)

            eps_metadata.append({
                "episode_index": ep_idx,
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
            })

        all_eps_metadata.append(eps_metadata)

    # 合并所有 camera 的元数据
    num_episodes = len(all_eps_metadata[0])
    merged = []
    for ep_idx in range(num_episodes):
        ep_dict = {}
        for cam_meta in all_eps_metadata:
            ep_dict.update(cam_meta[ep_idx])
        merged.append(ep_dict)

    return merged


def flatten_stats(stats: dict, prefix: str = "stats/") -> dict:
    """将 stats dict 展平为 key-value 对。"""
    flat = {}
    for stat_name, values in stats.items():
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if isinstance(values, (list, tuple)):
            for i, v in enumerate(values):
                flat[f"{prefix}{stat_name}/{i}"] = v
        else:
            flat[f"{prefix}{stat_name}"] = values
    return flat


def convert_episodes_metadata(
    root: Path,
    data_metadata: list[dict],
    video_metadata: list[dict] | None,
) -> None:
    """生成 meta/episodes/chunk-000/episodes.parquet"""
    # 读取旧 episodes
    episodes = load_jsonlines(root / "meta" / "episodes.jsonl")
    episodes_map = {e["episode_index"]: e for e in episodes}

    # 读取旧 episodes_stats
    episodes_stats_list = load_jsonlines(root / "meta" / "episodes_stats.jsonl")
    stats_map = {e["episode_index"]: e["stats"] for e in episodes_stats_list}

    rows = []
    for dm in data_metadata:
        ep_idx = dm["episode_index"]
        ep_info = episodes_map[ep_idx]
        ep_stats = stats_map.get(ep_idx, {})

        row = {**dm}
        # 视频元数据
        if video_metadata:
            vm = video_metadata[ep_idx]
            row.update(vm)
        # episodes 元数据
        row["tasks"] = ep_info.get("tasks", [])
        row["length"] = ep_info["length"]
        # 展平 stats
        flat_stats = flatten_stats(ep_stats)
        row.update(flat_stats)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 写入 episodes.parquet（按 chunk 分组）
    episodes_dir = root / "meta" / "episodes"
    for chunk_idx in sorted(df["data/chunk_index"].unique()):
        chunk_df = df[df["data/chunk_index"] == chunk_idx].reset_index(drop=True)
        out_dir = episodes_dir / f"chunk-{chunk_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "episodes.parquet"
        chunk_df.to_parquet(out_path, index=False)

    print(f"  episodes.parquet: {episodes_dir} ({len(df)} episodes)")


def convert_tasks(root: Path) -> None:
    """生成 meta/tasks.parquet（starVLA 期望的单文件路径）。"""
    tasks = load_jsonlines(root / "meta" / "tasks.jsonl")
    tasks = sorted(tasks, key=lambda x: x["task_index"])

    df = pd.DataFrame({
        "task_index": [t["task_index"] for t in tasks],
        "task": [t["task"] for t in tasks],
    }).set_index("task")

    out_path = root / "meta" / "tasks.parquet"
    df.to_parquet(out_path, index=True)
    print(f"  tasks.parquet: {out_path} ({len(df)} tasks)")


def convert_info(root: Path) -> None:
    """更新 meta/info.json"""
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["codebase_version"] = V30
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if get_video_keys(root) else None

    # 给非 video feature 加 fps 字段
    fps = info.get("fps", 30)
    info["fps"] = int(fps)
    for key, ft in info["features"].items():
        if ft["dtype"] not in ("video",):
            ft["fps"] = int(fps)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  info.json: codebase_version → {V30}")


def cleanup_old_files(root: Path) -> None:
    """删除 v2.1 旧文件"""
    old_files = [
        root / "meta" / "episodes.jsonl",
        root / "meta" / "episodes_stats.jsonl",
        root / "meta" / "tasks.jsonl",
    ]
    for f in old_files:
        if f.exists():
            f.unlink()
            print(f"  删除旧文件: {f}")

    # 删除旧的 videos/chunk-000/ 目录（已移到 videos/CAMERA/chunk-000/）
    old_video_chunks = root / "videos" / "chunk-000"
    if old_video_chunks.exists():
        shutil.rmtree(old_video_chunks)
        print(f"  删除旧视频目录: {old_video_chunks}")


def main():
    parser = argparse.ArgumentParser(description="LeRobot v2.1 → v3.0 本地转换")
    parser.add_argument("--dataset_dir", required=True, help="数据集根目录")
    parser.add_argument("--dry_run", action="store_true", help="只打印不执行")
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集不存在: {root}")

    validate_version(root)

    print(f"📦 转换 LeRobot v2.1 → v3.0: {root}")
    print()

    # 1. 转换 data parquet
    print("[1/6] 转换 data parquet ...")
    data_metadata = convert_data(root)
    print(f"  ✓ {len(data_metadata)} episodes")

    # 2. 转换 videos
    print("[2/6] 转换 videos ...")
    video_metadata = convert_videos(root)
    if video_metadata:
        print(f"  ✓ {len(video_metadata)} episodes × {len(get_video_keys(root))} cameras")
    else:
        print("  ✓ 无视频")

    # 3. 生成 episodes.parquet
    print("[3/6] 生成 episodes.parquet ...")
    convert_episodes_metadata(root, data_metadata, video_metadata)

    # 4. 生成 tasks.parquet
    print("[4/6] 生成 tasks.parquet ...")
    convert_tasks(root)

    # 5. 更新 info.json
    print("[5/6] 更新 info.json ...")
    convert_info(root)

    # 6. 清理旧文件
    print("[6/6] 清理旧文件 ...")
    cleanup_old_files(root)

    print(f"\n✅ 转换完成: {root}")

    # 验证
    episodes_parquet = list(root.glob("meta/episodes/*/*.parquet"))
    print(f"  meta/episodes/*/*.parquet: {len(episodes_parquet)} files")
    tasks_parquet = list(root.glob("meta/tasks/*/*.parquet"))
    print(f"  meta/tasks/*/*.parquet: {len(tasks_parquet)} files")

    # 删除之前可能残留的 steps_data_index.pkl（需要重新生成）
    steps_cache = root / "meta" / "steps_data_index.pkl"
    if steps_cache.exists():
        steps_cache.unlink()
        print(f"  删除旧缓存: {steps_cache}")


if __name__ == "__main__":
    main()
