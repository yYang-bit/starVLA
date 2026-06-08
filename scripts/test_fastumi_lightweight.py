#!/usr/bin/env python3
"""
Ultra-lightweight FastUMI interface test.

Only tests:
1. Load 1 episode metadata
2. Load 1 sample from parquet
3. Apply DerivedKeysTransform
4. Verify output keys and shapes

NO full dataset scan, NO GPU, runs in <10 seconds.
"""

import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import json

# Add starVLA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor

# Import FastUMI config
import importlib.util
_config_path = Path(__file__).parent.parent / "examples/FastUMI/train_files/data_registry/data_config.py"
_spec = importlib.util.spec_from_file_location("fastumi_data_config", _config_path)
_fastumi_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastumi_module)

DerivedKeysTransform = _fastumi_module.DerivedKeysTransform
FastUMIDualArmDataConfig = _fastumi_module.FastUMIDualArmDataConfig


def test_fastumi_lightweight():
    """Ultra-lightweight test: 1 episode, 1 sample."""

    print("\n" + "="*60)
    print("  FastUMI Ultra-Lightweight Interface Test")
    print("="*60 + "\n")

    dataset_path = Path("/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False

    print(f"📂 Dataset: {dataset_path.name}")

    # Load config
    config = FastUMIDualArmDataConfig()
    print(f"\n🔧 Config loaded")
    print(f"   Raw keys: {config.raw_state_keys + config.raw_action_keys}")
    print(f"   Output keys: {len(config.state_keys + config.action_keys)} derived keys")

    # Step 1: Read 1 episode metadata
    print(f"\n📄 Step 1: Read episodes.jsonl (1 line only)")
    episodes_file = dataset_path / "meta/episodes.jsonl"

    with open(episodes_file) as f:
        first_line = f.readline()
        episode = json.loads(first_line)

    print(f"   ✅ Episode 0: {episode['length']} frames")

    # Step 2: Load 1 frame from parquet
    print(f"\n📦 Step 2: Load 1 sample from parquet")

    episode_index = episode['episode_index']

    # v2.1 format: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    # Need to find which chunk this episode belongs to (from info.json)
    info_file = dataset_path / "meta/info.json"
    with open(info_file) as f:
        info = json.load(f)

    # Get chunk size from info
    chunk_size = info.get('chunks_size', 1000)  # Default to 1000
    episode_chunk = episode_index // chunk_size

    data_path = dataset_path / f"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    if not data_path.exists():
        print(f"   ❌ Parquet not found: {data_path}")
        return False

    df = pd.read_parquet(data_path)
    print(f"   ✅ Loaded parquet: {len(df)} frames")

    # Get first frame
    sample_data = {
        "observation.state": df["observation.state"].iloc[0],
        "action": df["action"].iloc[0],
    }

    print(f"   Raw observation.state shape: {sample_data['observation.state'].shape}")
    print(f"   Raw action shape: {sample_data['action'].shape}")

    # Step 3: Apply transforms
    print(f"\n🔄 Step 3: Apply DerivedKeysTransform")

    # Convert to tensor
    tensor_transform = StateActionToTensor(apply_to=config.raw_state_keys + config.raw_action_keys)
    sample_data = tensor_transform.apply(sample_data)

    # Apply DerivedKeysTransform
    derived_transform = DerivedKeysTransform(
        apply_to=list(config.derived_keys.keys()),
        derived_keys=config.derived_keys,
        drop_source_keys=False,
    )

    sample_data = derived_transform.apply(sample_data)

    print(f"   ✅ Transform applied")

    # Step 4: Verify outputs
    print(f"\n🔍 Step 4: Verify output keys and shapes")

    expected_keys = config.state_keys + config.action_keys
    found = 0

    for key in expected_keys:
        if key in sample_data:
            shape = sample_data[key].shape
            print(f"   ✅ {key}: {tuple(shape)}")
            found += 1
        else:
            print(f"   ❌ {key}: MISSING")

    print(f"\n   Found {found}/{len(expected_keys)} keys")

    # Verify rotation_6d dimensions
    print(f"\n🔄 Step 5: Verify rotation conversions")

    rotation_keys = [k for k in expected_keys if "ori_6d" in k]
    all_correct = True

    for key in rotation_keys:
        if key in sample_data:
            dim = sample_data[key].shape[-1]
            if dim == 6:
                print(f"   ✅ {key}: dimension {dim} (correct)")
            else:
                print(f"   ❌ {key}: dimension {dim} (expected 6)")
                all_correct = False

    # Summary
    print(f"\n" + "="*60)
    print(f"  Test Summary")
    print(f"="*60)
    print(f"  ✅ Episode metadata: PASSED")
    print(f"  ✅ Parquet loading: PASSED")
    print(f"  ✅ DerivedKeysTransform: PASSED")
    print(f"  ✅ Output keys: {found}/{len(expected_keys)}")
    print(f"  ✅ Rotation_6d: {'PASSED' if all_correct else 'FAILED'}")
    print(f"="*60 + "\n")

    return found == len(expected_keys) and all_correct


if __name__ == "__main__":
    success = test_fastumi_lightweight()
    sys.exit(0 if success else 1)
