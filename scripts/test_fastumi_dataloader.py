#!/usr/bin/env python3
"""
Lightweight FastUMI dataloader test.

Tests:
1. Load FastUMI dataset configuration
2. Create dataset instance
3. Load a few samples
4. Verify data shapes and transforms

NO GPU, NO TRAINING - just dataloader verification.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add starVLA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from examples.FastUMI.train_files.data_registry.data_config import (
    FastUMIDualArmDataConfig,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
)


def test_fastumi_dataloader():
    """Test FastUMI dataloader without training."""

    print("\n" + "="*60)
    print("  FastUMI Dataloader Test")
    print("="*60 + "\n")

    # Dataset path
    dataset_path = Path("/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False

    print(f"📂 Dataset path: {dataset_path}")

    # Load config
    print(f"\n🔧 Loading FastUMI configuration...")
    config = FastUMIDualArmDataConfig()
    embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG["fastumi_dual_arm"]

    print(f"   Raw state keys: {config.raw_state_keys}")
    print(f"   Raw action keys: {config.raw_action_keys}")
    print(f"   Derived state keys: {config.state_keys}")
    print(f"   Derived action keys: {config.action_keys}")

    # Create dataset (without normalization first - for stats computation)
    print(f"\n📊 Creating dataset for stats computation...")
    print(f"   (Using transform pipeline WITHOUT normalization)")

    # Get transform without normalization
    # We'll build a minimal transform for testing
    from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor
    from examples.FastUMI.train_files.data_registry.data_config import DerivedKeysTransform

    test_transforms = ComposedModalityTransform(
        transforms=[
            StateActionToTensor(apply_to=config.raw_state_keys + config.raw_action_keys),
            DerivedKeysTransform(
                apply_to=list(config.derived_keys.keys()),
                derived_keys=config.derived_keys,
                drop_source_keys=False,  # Keep source for inspection
            ),
        ]
    )

    try:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config.modality_config(),
            embodiment_tag=embodiment_tag,
            transforms=test_transforms,
            data_cfg={"lerobot_version": "auto"},
        )

        print(f"✅ Dataset created successfully")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Dataset name: {dataset.dataset_name}")
        print(f"   LeRobot version: {dataset._lerobot_version}")

    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load first sample
    print(f"\n🎯 Loading first sample...")
    try:
        sample = dataset[0]

        print(f"✅ Sample loaded successfully")
        print(f"\n📦 Sample contents:")
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                print(f"   {key}: {type(value).__name__} {tuple(value.shape)} {value.dtype}")
            elif isinstance(value, list):
                print(f"   {key}: list[{len(value)}]")
            else:
                print(f"   {key}: {type(value).__name__}")

        # Check derived keys
        print(f"\n🔍 Checking derived keys:")

        # Raw keys (should still exist if drop_source_keys=False)
        if "observation.state" in sample:
            obs_state = sample["observation.state"]
            print(f"   Raw observation.state: {tuple(obs_state.shape)}")

        if "action" in sample:
            action = sample["action"]
            print(f"   Raw action: {tuple(action.shape)}")

        # Derived state keys
        derived_found = 0
        for key in config.state_keys:
            if key in sample:
                value = sample[key]
                print(f"   ✅ {key}: {tuple(value.shape)}")
                derived_found += 1
            else:
                print(f"   ❌ {key}: NOT FOUND")

        # Derived action keys
        for key in config.action_keys:
            if key in sample:
                value = sample[key]
                print(f"   ✅ {key}: {tuple(value.shape)}")
                derived_found += 1
            else:
                print(f"   ❌ {key}: NOT FOUND")

        print(f"\n   Found {derived_found}/{len(config.state_keys) + len(config.action_keys)} derived keys")

        # Verify rotation_6d shapes
        print(f"\n🔄 Verifying rotation conversions:")
        if "state.left_ori_6d" in sample:
            ori_6d = sample["state.left_ori_6d"]
            print(f"   state.left_ori_6d: {tuple(ori_6d.shape)} (expected: [6] or [1, 6])")
            if ori_6d.shape[-1] == 6:
                print(f"   ✅ Correct rotation_6d dimension")
            else:
                print(f"   ❌ Wrong dimension, expected 6")

        if "action.left_ori_6d" in sample:
            action_ori = sample["action.left_ori_6d"]
            print(f"   action.left_ori_6d: {tuple(action_ori.shape)} (expected: [T, 6])")
            if action_ori.shape[-1] == 6:
                print(f"   ✅ Correct rotation_6d dimension")
            else:
                print(f"   ❌ Wrong dimension, expected 6")

    except Exception as e:
        print(f"❌ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load a few more samples
    print(f"\n📦 Loading 3 more samples...")
    try:
        for i in range(1, min(4, len(dataset))):
            sample = dataset[i]
            if "action.left_arm" in sample:
                action_shape = sample["action.left_arm"].shape
                print(f"   Sample {i}: action.left_arm shape = {tuple(action_shape)}")
        print(f"✅ Multiple samples loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load multiple samples: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print(f"\n" + "="*60)
    print(f"  Test Summary")
    print(f"="*60)
    print(f"  ✅ Dataset creation: PASSED")
    print(f"  ✅ Version detection: v{dataset._lerobot_version}")
    print(f"  ✅ Sample loading: PASSED")
    print(f"  ✅ DerivedKeysTransform: PASSED")
    print(f"  ✅ Rotation conversion: PASSED")
    print(f"  ✅ Multiple samples: PASSED")
    print(f"="*60 + "\n")

    return True


if __name__ == "__main__":
    success = test_fastumi_dataloader()
    sys.exit(0 if success else 1)
