#!/usr/bin/env python3
"""
Universal dataset statistics computation tool.

This script works with any dataset configuration by:
1. Loading the dataset-specific config
2. Using the config's transform pipeline (without normalization)
3. Computing statistics over the transformed data
4. Saving to the specified output path

Usage:
    python compute_dataset_stats.py \
        --dataset_path /path/to/data \
        --robot_type galbot_bimanual \
        --data_registry examples.Galbot.train_files.data_registry.data_config \
        --output /path/to/data/meta/stats.json
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add starVLA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def _read_all_parquet_data(dataset_path: Path) -> dict:
    """Read all parquet data, grouped by trajectory_id.
    Bypasses video decoding entirely.

    Returns:
        dict: {trajectory_id: DataFrame}
    """
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LE_ROBOT3_EPISODE_FILENAME,
        LE_ROBOT2_EPISODES_FILENAME,
        LE_ROBOT2_DEFAULT_DATA_PATH,
    )
    import pandas as pd

    dataset_path = Path(dataset_path)
    trajectories = {}

    # Detect version
    v21_episodes = dataset_path / LE_ROBOT2_EPISODES_FILENAME
    if v21_episodes.exists():
        # v2.1: read episodes.jsonl, load each parquet
        import json
        with open(v21_episodes) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                eid = int(ep.get("episode_index", 0))
                chunk = int(ep.get("episode_chunk", eid // 1000))
                pq_path = dataset_path / LE_ROBOT2_DEFAULT_DATA_PATH.format(
                    episode_chunk=chunk, episode_index=eid
                )
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                    if "episode_index" in df.columns:
                        df = df.loc[df["episode_index"] == eid].copy()
                    trajectories[eid] = df.reset_index(drop=True)
    else:
        # v3.0: read all parquet files under data/
        data_files = sorted(dataset_path.glob("data/*/*.parquet"))
        for pq_path in data_files:
            df = pd.read_parquet(pq_path)
            if "episode_index" in df.columns:
                for eid, sub_df in df.groupby("episode_index"):
                    trajectories[int(eid)] = sub_df.reset_index(drop=True)
            else:
                trajectories[len(trajectories)] = df.reset_index(drop=True)

    return trajectories


def _build_raw_sample(traj_df, base_idx: int, chunk_size: int, modality_configs: dict, modality_meta=None):
    """Build a raw sample dict from a trajectory DataFrame.

    Slices flat parquet vectors (e.g. 'action' [20]) into structured subkeys
    (e.g. 'action.left_pos' [3]) using modality_meta (from modality.json).

    For 'action' modality: builds a chunk [chunk_size, D] starting at base_idx.
    For 'state' modality: takes the single frame at base_idx.
    For 'video'/'language': skipped (not needed for stats).

    Args:
        modality_meta: LeRobotModalityMetadata object (maps subkey -> original_key + start/end)

    Returns:
        dict: raw sample with subkey entries, or None if out of range.
    """
    import numpy as np

    traj_len = len(traj_df)
    sample = {}

    if modality_meta is None:
        return sample

    for modality, cfg in modality_configs.items():
        if modality in ("video", "language"):
            continue

        modality_meta_obj = getattr(modality_meta, modality, None)
        if modality_meta_obj is None:
            continue

        # modality_meta_obj is a dict: {subkey: LeRobotStateActionMetadata}
        # modality_keys are subkeys (e.g. 'action.left_pos')
        for subkey in cfg.modality_keys:
            # strip modality prefix to get the meta subkey
            meta_subkey = subkey.split(".", 1)[1] if "." in subkey else subkey

            field_meta = None
            if isinstance(modality_meta_obj, dict):
                field_meta = modality_meta_obj.get(meta_subkey)
            else:
                field_meta = getattr(modality_meta_obj, meta_subkey, None)

            if field_meta is None:
                continue

            original_key = field_meta.original_key
            start = field_meta.start
            end = field_meta.end

            if original_key not in traj_df.columns:
                continue

            # Get the full column as array [traj_len, D]
            col_values = np.stack([np.asarray(v, dtype=np.float32) for v in traj_df[original_key].values])
            # Slice to [start:end]
            col_values = col_values[:, start:end]

            if modality == "action":
                # Build chunk [chunk_size, D]
                end_idx = base_idx + chunk_size
                if end_idx > traj_len:
                    chunk = np.zeros((chunk_size, col_values.shape[-1]), dtype=np.float32)
                    valid = traj_len - base_idx
                    chunk[:valid] = col_values[base_idx:traj_len]
                    chunk[valid:] = col_values[-1]
                else:
                    chunk = col_values[base_idx:end_idx].copy()
                sample[subkey] = chunk
            else:
                # state: single frame at base_idx
                sample[subkey] = col_values[base_idx].copy()

    return sample


def compute_statistics(
    dataset_path: Path,
    robot_type: str,
    data_registry_module: str,
    output_path: Path,
    sample_ratio: float = 1.0,
    action_keys_only: bool = True,
    action_mode: str = None,
    action_chunk_size: int = None,
    gripper_normalization: str = None,
):
    """Compute dataset statistics.

    Args:
        dataset_path: Path to dataset directory
        robot_type: Robot type key (e.g., "galbot_bimanual", "fastumi_dual_arm")
        data_registry_module: Python module path to data_config.py
        output_path: Where to save stats JSON
        sample_ratio: Fraction of dataset to use (1.0 = all data)
        action_keys_only: Only compute stats for action keys (default True)
    """
    print(f"\n{'='*60}")
    print(f"  Computing Statistics for {robot_type}")
    print(f"{'='*60}\n")

    # 1. Load configuration
    print(f"📦 Loading config from {data_registry_module}...")
    try:
        registry = import_module(data_registry_module)
        config = registry.ROBOT_TYPE_CONFIG_MAP[robot_type]
        embodiment_tag = registry.ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    except (ImportError, KeyError, AttributeError) as e:
        print(f"❌ Failed to load config: {e}")
        print(f"\nMake sure:")
        print(f"  1. Module exists: {data_registry_module}")
        print(f"  2. ROBOT_TYPE_CONFIG_MAP contains '{robot_type}'")
        print(f"  3. ROBOT_TYPE_TO_EMBODIMENT_TAG contains '{robot_type}'")
        sys.exit(1)

    print(f"✅ Config loaded for {robot_type}")

    # 2. Get transform pipeline WITHOUT normalization
    print(f"\n🔧 Building transform pipeline (without normalization)...")

    # Build data_cfg from parameters
    data_cfg = {}
    if action_mode is not None:
        data_cfg["action_mode"] = action_mode
    if action_chunk_size is not None:
        data_cfg["action_chunk_size"] = action_chunk_size
    if gripper_normalization is not None:
        data_cfg["gripper_normalization"] = gripper_normalization

    print(f"   data_cfg: {data_cfg}")

    # Check if config has a special method for stats computation
    if hasattr(config, 'transform_for_stats'):
        transforms = config.transform_for_stats(data_cfg)
        print(f"   Using config.transform_for_stats(data_cfg)")
    elif hasattr(config, 'transform_pipeline_without_normalization'):
        transforms = config.transform_pipeline_without_normalization()
        print(f"   Using config.transform_pipeline_without_normalization()")
    else:
        # Use default transform but warn user
        transforms = config.transform(data_cfg)
        print(f"⚠️  Using default transform - may include normalization!")
        print(f"   Consider adding transform_for_stats() method to config")

    # 3. Get modality configs to know how to slice flat parquet vectors into subkeys
    modality_configs = config.modality_config(data_cfg)
    action_chunk_size = int(data_cfg.get("action_chunk_size", 30))
    print(f"   action_chunk_size: {action_chunk_size}")

    # 3.5 Load modality metadata (maps subkey -> original_key + start/end)
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LE_ROBOT_MODALITY_FILENAME,
        LE_ROBOT_INFO_FILENAME,
        _load_lerobot_modality_metadata,
        _build_direct_lerobot_modality_metadata,
    )
    modality_meta_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    info_meta_path = dataset_path / LE_ROBOT_INFO_FILENAME
    if modality_meta_path.exists():
        modality_meta = _load_lerobot_modality_metadata(modality_meta_path, info_meta_path)
        print(f"   Loaded modality metadata from {modality_meta_path}")
    else:
        with open(info_meta_path) as f:
            info_meta = json.load(f)
        modality_meta = _build_direct_lerobot_modality_metadata(modality_configs, info_meta)
        print(f"   Built modality metadata from info.json")

    # 4. Read parquet data directly (NO video decoding)
    print(f"\n📂 Reading parquet data directly from {dataset_path}...")
    try:
        parquet_data = _read_all_parquet_data(dataset_path)
        print(f"✅ Read {len(parquet_data)} trajectories from parquet")
    except Exception as e:
        print(f"❌ Failed to read parquet: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine which keys to compute stats for
    if action_keys_only:
        if hasattr(config, 'action_keys'):
            keys_to_compute = list(config.action_keys)
            # Include normalized state keys when present (e.g., dual_relative_pose_pos)
            # so training can load state statistics from the same stats cache.
            state_norm_keys = []
            if hasattr(config, 'state_keys'):
                # By convention rot6d keys are not normalized; pos keys usually are.
                state_norm_keys = [k for k in config.state_keys if k.endswith('_pos')]
            keys_to_compute.extend(state_norm_keys)
        else:
            keys_to_compute = None
    else:
        keys_to_compute = None

    print(f"\n📊 Computing statistics:")
    if keys_to_compute:
        print(f"   Keys: {keys_to_compute}")
    else:
        print(f"   Keys: All modalities")

    # 5. Sample trajectories (not samples)
    all_traj_ids = list(parquet_data.keys())
    if sample_ratio < 1.0:
        num_trajs = max(1, int(len(all_traj_ids) * sample_ratio))
        print(f"\n🎲 Sampling {sample_ratio*100:.1f}% of trajectories ({num_trajs}/{len(all_traj_ids)})")
        sampled_traj_ids = np.random.choice(all_traj_ids, num_trajs, replace=False).tolist()
    else:
        sampled_traj_ids = all_traj_ids

    # 6. Collect data: slice flat parquet vectors into subkeys, build chunks, apply transform
    print(f"\n🔄 Collecting data from {len(sampled_traj_ids)} trajectories...")

    all_data = {}  # {key: list of arrays}

    for traj_id in tqdm(sampled_traj_ids, desc="Processing trajectories"):
        traj_df = parquet_data[traj_id]
        traj_len = len(traj_df)

        # Sample base indices within this trajectory (step = action_chunk_size to avoid overlap)
        if traj_len <= action_chunk_size:
            continue
        num_chunks = max(1, traj_len // action_chunk_size)
        base_indices = np.linspace(0, traj_len - action_chunk_size - 1, num_chunks, dtype=int)

        for base_idx in base_indices:
            try:
                # Build a raw sample dict from parquet (NO video)
                raw_sample = _build_raw_sample(
                    traj_df, base_idx, action_chunk_size, modality_configs, modality_meta
                )
                if raw_sample is None or len(raw_sample) == 0:
                    continue

                # Apply transform pipeline (ActionChunkTransform etc., no normalization)
                sample = transforms(raw_sample)

                # Extract data for keys_to_compute
                if keys_to_compute is not None:
                    for key in keys_to_compute:
                        if key in sample:
                            value = sample[key]
                            if isinstance(value, torch.Tensor):
                                value = value.detach().cpu().numpy()
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].append(np.asarray(value, dtype=np.float32))
                else:
                    # Use flat keys
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().numpy()
                        if not isinstance(value, np.ndarray):
                            continue
                        if key not in all_data:
                            all_data[key] = []
                        all_data[key].append(np.asarray(value, dtype=np.float32))

            except Exception as e:
                continue

    if not all_data:
        print(f"\n❌ No data collected! Check transform pipeline and sample extraction.")
        sys.exit(1)

    print(f"✅ Collected data for {len(all_data)} keys")

    # 7. Compute statistics
    print(f"\n📈 Computing statistics...")
    stats = {}

    for key, data_list in all_data.items():
        print(f"   {key}:")

        # Concatenate all samples
        try:
            concatenated = np.concatenate(data_list, axis=0).astype(np.float32)
        except Exception as e:
            print(f"      ⚠️  Failed to concatenate: {e}")
            continue

        print(f"      Shape: {concatenated.shape}")

        # Ensure 2D [N, D] before statistics; single-frame state vectors may
        # concatenate to [N*D], so reshape them using the first sample width.
        if concatenated.ndim == 1:
            first_width = np.asarray(data_list[0]).reshape(-1).shape[-1]
            if first_width > 1 and concatenated.size % first_width == 0:
                concatenated = concatenated.reshape(-1, first_width)
            else:
                concatenated = concatenated.reshape(-1, 1)

        # Compute stats
        key_stats = {
            "mean": np.asarray(np.mean(concatenated, axis=0)).reshape(-1).tolist(),
            "std": np.asarray(np.std(concatenated, axis=0)).reshape(-1).tolist(),
            "min": np.asarray(np.min(concatenated, axis=0)).reshape(-1).tolist(),
            "max": np.asarray(np.max(concatenated, axis=0)).reshape(-1).tolist(),
            "q01": np.asarray(np.quantile(concatenated, 0.01, axis=0)).reshape(-1).tolist(),
            "q99": np.asarray(np.quantile(concatenated, 0.99, axis=0)).reshape(-1).tolist(),
        }

        # Print summary
        print(f"      Mean: [{key_stats['mean'][0]:.3f}, ..., {key_stats['mean'][-1]:.3f}]")
        print(f"      Std:  [{key_stats['std'][0]:.3f}, ..., {key_stats['std'][-1]:.3f}]")
        print(f"      Q01:  [{key_stats['q01'][0]:.3f}, ..., {key_stats['q01'][-1]:.3f}]")
        print(f"      Q99:  [{key_stats['q99'][0]:.3f}, ..., {key_stats['q99'][-1]:.3f}]")

        stats[key] = key_stats

    # 8. Save to file
    print(f"\n💾 Saving statistics to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Statistics saved!")
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_path.name}")
    print(f"  Trajectories sampled: {len(sampled_traj_ids)} ({sample_ratio*100:.1f}% of total)")
    print(f"  Keys: {len(stats)}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics using dataset-specific config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Galbot dataset with delta mode
  python compute_dataset_stats.py \\
      --dataset_path /path/to/galbot_data \\
      --robot_type galbot_bimanual_self \\
      --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \\
      --output /path/to/galbot_data/meta/stats_delta_chunk30.json \\
      --action_mode delta \\
      --action_chunk_size 30

  # FastUMI dataset with relative_pose mode (large dataset, use sampling)
  python compute_dataset_stats.py \\
      --dataset_path /path/to/fastumi_data/Add_Rice_to_Rice_Cooker \\
      --robot_type fastumi_dual_arm \\
      --data_registry examples.FastUMI.train_files.data_registry.data_config \\
      --output /path/to/fastumi_data/Add_Rice_to_Rice_Cooker/meta/stats_relative_pose_chunk16.json \\
      --action_mode relative_pose \\
      --action_chunk_size 16 \\
      --sample_ratio 0.1
        """
    )

    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to dataset directory (contains meta/ and data/)"
    )

    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        help="Robot type key from ROBOT_TYPE_CONFIG_MAP (e.g., 'galbot_bimanual')"
    )

    parser.add_argument(
        "--data_registry",
        type=str,
        required=True,
        help="Python module path to data_config.py (e.g., 'examples.Galbot.train_files.data_registry.data_config')"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for stats JSON file"
    )

    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0, default: 1.0)"
    )

    parser.add_argument(
        "--all_keys",
        action="store_true",
        help="Compute stats for all keys, not just actions"
    )

    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        help="Action mode: abs, delta, or relative_pose (default: use config default)"
    )

    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=None,
        help="Action chunk size (default: use config default)"
    )

    parser.add_argument(
        "--gripper_normalization",
        type=str,
        default=None,
        choices=["binary", "min_max", "none"],
        help="Gripper normalization method (default: None)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.dataset_path.exists():
        print(f"❌ Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    if not 0.0 < args.sample_ratio <= 1.0:
        print(f"❌ sample_ratio must be between 0.0 and 1.0, got {args.sample_ratio}")
        sys.exit(1)

    # Run computation
    compute_statistics(
        dataset_path=args.dataset_path,
        robot_type=args.robot_type,
        data_registry_module=args.data_registry,
        output_path=args.output,
        sample_ratio=args.sample_ratio,
        action_keys_only=not args.all_keys,
        action_mode=args.action_mode,
        action_chunk_size=args.action_chunk_size,
        gripper_normalization=args.gripper_normalization if args.gripper_normalization != "none" else None,
    )


if __name__ == "__main__":
    main()
