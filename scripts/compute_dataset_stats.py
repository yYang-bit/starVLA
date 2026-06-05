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


def compute_statistics(
    dataset_path: Path,
    robot_type: str,
    data_registry_module: str,
    output_path: Path,
    sample_ratio: float = 1.0,
    action_keys_only: bool = True,
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

    # Check if config has a special method for stats computation
    if hasattr(config, 'transform_for_stats'):
        transforms = config.transform_for_stats()
        print(f"   Using config.transform_for_stats()")
    elif hasattr(config, 'transform_pipeline_without_normalization'):
        transforms = config.transform_pipeline_without_normalization()
        print(f"   Using config.transform_pipeline_without_normalization()")
    else:
        # Use default transform but warn user
        transforms = config.transform(data_cfg={})
        print(f"⚠️  Using default transform - may include normalization!")
        print(f"   Consider adding transform_for_stats() method to config")

    # 3. Create dataset
    print(f"\n📂 Loading dataset from {dataset_path}...")
    try:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config.modality_config(),
            embodiment_tag=embodiment_tag,
            transforms=transforms,
            data_cfg={"lerobot_version": "auto"},  # Auto-detect version
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine which keys to compute stats for
    if action_keys_only:
        if hasattr(config, 'action_keys'):
            keys_to_compute = config.action_keys
        else:
            # Fallback: use all action.* keys from modality_config
            action_config = config.modality_config().get('action')
            if action_config:
                keys_to_compute = action_config.output_keys or action_config.modality_keys
            else:
                print(f"⚠️  Could not find action keys, computing for all modalities")
                keys_to_compute = None
    else:
        keys_to_compute = None  # Compute for all keys

    print(f"\n📊 Computing statistics:")
    if keys_to_compute:
        print(f"   Keys: {keys_to_compute}")
    else:
        print(f"   Keys: All modalities")

    # 5. Sample dataset
    num_samples = int(len(dataset) * sample_ratio)
    if sample_ratio < 1.0:
        print(f"\n🎲 Sampling {sample_ratio*100:.1f}% of data ({num_samples} samples)")
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = range(len(dataset))

    # 6. Collect data
    print(f"\n🔄 Collecting data from {num_samples} samples...")

    all_data = {}  # {key: list of arrays}

    for idx in tqdm(indices, desc="Processing samples"):
        try:
            sample = dataset[idx]

            # Extract action data
            if 'action' in sample:
                action = sample['action']
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()

                # If keys_to_compute is specified, we assume action is already structured
                # Otherwise, use the full action tensor
                if keys_to_compute is None:
                    # Flatten if multi-dimensional (e.g., [T, D] -> [T*D])
                    if action.ndim > 1:
                        action = action.reshape(-1, action.shape[-1])
                    if 'action' not in all_data:
                        all_data['action'] = []
                    all_data['action'].append(action)
                else:
                    # This assumes sample contains structured keys already
                    # (after DerivedKeysTransform or similar)
                    for key in keys_to_compute:
                        if key in sample:
                            value = sample[key]
                            if isinstance(value, torch.Tensor):
                                value = value.detach().cpu().numpy()
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].append(value)

                    # If keys not found in sample, fall back to action tensor
                    if not all_data:
                        print(f"⚠️  Structured keys not found in sample, using flat action")
                        if action.ndim > 1:
                            action = action.reshape(-1, action.shape[-1])
                        if 'action' not in all_data:
                            all_data['action'] = []
                        all_data['action'].append(action)
                        keys_to_compute = None

        except Exception as e:
            print(f"\n⚠️  Skipping sample {idx}: {e}")
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

        # Compute stats
        key_stats = {
            "mean": np.mean(concatenated, axis=0).tolist(),
            "std": np.std(concatenated, axis=0).tolist(),
            "min": np.min(concatenated, axis=0).tolist(),
            "max": np.max(concatenated, axis=0).tolist(),
            "q01": np.quantile(concatenated, 0.01, axis=0).tolist(),
            "q99": np.quantile(concatenated, 0.99, axis=0).tolist(),
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
    print(f"  Samples: {num_samples} ({sample_ratio*100:.1f}% of total)")
    print(f"  Keys: {len(stats)}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics using dataset-specific config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Galbot dataset
  python compute_dataset_stats.py \\
      --dataset_path /path/to/galbot_data \\
      --robot_type galbot_bimanual \\
      --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \\
      --output /path/to/galbot_data/meta/stats_delta_chunk30.json

  # FastUMI dataset (large, use sampling)
  python compute_dataset_stats.py \\
      --dataset_path /path/to/fastumi_data/Add_Rice_to_Rice_Cooker \\
      --robot_type fastumi_dual_arm \\
      --data_registry examples.FastUMI.train_files.data_registry.data_config \\
      --output /path/to/fastumi_data/Add_Rice_to_Rice_Cooker/meta/stats.json \\
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
    )


if __name__ == "__main__":
    main()
