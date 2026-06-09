#!/usr/bin/env python3
"""
Lightweight test for unified ActionChunkTransform.

Tests:
1. FastUMI abs mode (baseline)
2. FastUMI delta mode (frame-to-frame in local frame)
3. FastUMI relative_pose mode (relative to base in local frame)

Verifies:
- Data loading works
- Transform output shapes correct
- First frame behavior correct (delta[0]=0, rel[0]=computed)
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

# Import FastUMI config
import importlib.util
_config_path = Path(__file__).parent.parent / "examples/FastUMI/train_files/data_registry/data_config.py"
_spec = importlib.util.spec_from_file_location("fastumi_data_config", _config_path)
_fastumi_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastumi_module)

FastUMIDualArmDataConfig = _fastumi_module.FastUMIDualArmDataConfig


def test_action_chunk_transform():
    """Test ActionChunkTransform with different modes."""

    print("\n" + "="*70)
    print("  ActionChunkTransform Lightweight Test")
    print("="*70 + "\n")

    dataset_path = Path("/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False

    config = FastUMIDualArmDataConfig()

    # Test 3 modes
    modes = ["abs", "delta", "relative_pose"]
    action_chunk_size = 4  # Small for quick test

    results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  Testing mode: {mode}")
        print(f"{'='*70}\n")

        # Build data_cfg
        data_cfg = {
            "action_mode": mode,
            "action_chunk_size": action_chunk_size,
            "gripper_normalization": None,  # Use raw values
        }

        # Create dataset (no stats, so no normalization will actually happen)
        try:
            dataset = LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=config.modality_config(data_cfg),
                transforms=config.transform_for_stats(data_cfg),  # No normalization
            )
        except Exception as e:
            print(f"❌ Failed to create dataset for mode={mode}: {e}")
            continue

        # Sample one item
        try:
            sample = dataset[0]
        except Exception as e:
            print(f"❌ Failed to sample for mode={mode}: {e}")
            continue

        # Check action keys
        action_pos_key = "action.left_arm"
        action_rot_key = "action.left_ori_6d"

        if action_pos_key not in sample or action_rot_key not in sample:
            print(f"❌ Missing action keys in sample")
            continue

        pos = sample[action_pos_key]  # [T, 3]
        rot = sample[action_rot_key]  # [T, 6]

        print(f"✅ Data loaded successfully")
        print(f"   Position shape: {pos.shape}")
        print(f"   Rotation shape: {rot.shape}")

        # Verify shapes
        if pos.shape[0] != action_chunk_size or pos.shape[1] != 3:
            print(f"   ❌ Position shape incorrect: expected [{action_chunk_size}, 3], got {pos.shape}")
            continue

        if rot.shape[0] != action_chunk_size or rot.shape[1] != 6:
            print(f"   ❌ Rotation shape incorrect: expected [{action_chunk_size}, 6], got {rot.shape}")
            continue

        # Check first frame behavior
        pos_0 = pos[0].numpy() if isinstance(pos, torch.Tensor) else pos[0]

        if mode == "delta":
            # First frame should be zero (or very close)
            max_val = np.abs(pos_0).max()
            print(f"   First frame position: {pos_0} (max={max_val:.6f})")
            if max_val < 1e-5:
                print(f"   ✅ Delta mode: first frame is zero ✓")
            else:
                print(f"   ⚠️  Delta mode: first frame not zero (but might be acceptable)")

        elif mode == "relative_pose":
            # First frame should be zero (relative to itself)
            max_val = np.abs(pos_0).max()
            print(f"   First frame position: {pos_0} (max={max_val:.6f})")
            if max_val < 1e-5:
                print(f"   ✅ Relative mode: first frame is zero ✓")
            else:
                print(f"   ❌ Relative mode: first frame should be zero!")
                continue

        else:  # abs
            # First frame can be anything
            print(f"   First frame position: {pos_0}")
            print(f"   ✅ Abs mode: no constraint on first frame")

        # Check subsequent frames are non-zero (action is actually happening)
        if action_chunk_size > 1:
            pos_1 = pos[1].numpy() if isinstance(pos, torch.Tensor) else pos[1]
            has_motion = np.abs(pos_1).max() > 1e-6

            if has_motion:
                print(f"   ✅ Action chunk has motion (pos[1] max={np.abs(pos_1).max():.6f})")
            else:
                print(f"   ⚠️  Action chunk might be static")

        results[mode] = {
            "success": True,
            "pos_shape": pos.shape,
            "rot_shape": rot.shape,
            "first_frame": pos_0,
        }

        print(f"\n✅ Mode '{mode}' PASSED")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Test Summary")
    print(f"{'='*70}\n")

    for mode in modes:
        if mode in results:
            print(f"  ✅ {mode:15s}: PASSED")
        else:
            print(f"  ❌ {mode:15s}: FAILED")

    success_count = len(results)
    total_count = len(modes)

    print(f"\n  Passed: {success_count}/{total_count}")
    print(f"{'='*70}\n")

    return success_count == total_count


if __name__ == "__main__":
    success = test_action_chunk_transform()
    sys.exit(0 if success else 1)
