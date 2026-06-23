#!/usr/bin/env python3
"""
Quick test for ActionChunkTransform - no data loading.
Tests transform logic directly with synthetic data.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.transform.action_chunk_mode import ActionChunkTransform


def test_transform_modes():
    """Test ActionChunkTransform with synthetic data."""

    print("\n" + "="*70)
    print("  ActionChunkTransform Logic Test (No Data Loading)")
    print("="*70 + "\n")

    # Create synthetic action chunk
    T = 4  # 4 frames
    action_keys = ["action.left_arm", "action.left_ori_6d", "action.left_gripper"]

    # Synthetic data
    data = {
        "action.left_arm": torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ], dtype=torch.float32),
        "action.left_ori_6d": torch.tensor([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32),
        "action.left_gripper": torch.tensor([
            [0.5],
            [0.6],
            [0.7],
            [0.8],
        ], dtype=torch.float32),
    }

    results = {}

    # Test abs mode
    print("Testing mode: abs")
    transform_abs = ActionChunkTransform(
        mode="abs",
        apply_to=action_keys,
        position_suffix="_arm",
        rotation_suffix="_ori_6d",
        gripper_suffix="_gripper",
    )
    data_abs = transform_abs.apply(data.copy())
    pos_abs = data_abs["action.left_arm"]
    print(f"  ✅ Abs mode: position unchanged (first frame = {pos_abs[0].tolist()})")
    results["abs"] = True

    # Test delta mode
    print("\nTesting mode: delta")
    transform_delta = ActionChunkTransform(
        mode="delta",
        apply_to=action_keys,
        position_suffix="_arm",
        rotation_suffix="_ori_6d",
        gripper_suffix="_gripper",
    )
    data_delta = transform_delta.apply(data.copy())
    pos_delta = data_delta["action.left_arm"]

    # Check first frame is zero
    first_frame_zero = torch.allclose(pos_delta[0], torch.zeros(3), atol=1e-5)
    print(f"  First frame: {pos_delta[0].tolist()}")
    if first_frame_zero:
        print(f"  ✅ Delta mode: first frame is zero")
        results["delta"] = True
    else:
        print(f"  ❌ Delta mode: first frame should be zero")
        results["delta"] = False

    # Check subsequent frames
    print(f"  Frame 1 delta: {pos_delta[1].tolist()}")
    print(f"  Frame 2 delta: {pos_delta[2].tolist()}")

    # Test relative_pose mode
    print("\nTesting mode: relative_pose")
    transform_rel = ActionChunkTransform(
        mode="relative_pose",
        apply_to=action_keys,
        position_suffix="_arm",
        rotation_suffix="_ori_6d",
        gripper_suffix="_gripper",
    )
    data_rel = transform_rel.apply(data.copy())
    pos_rel = data_rel["action.left_arm"]

    # Check first frame is zero (relative to itself)
    first_frame_zero = torch.allclose(pos_rel[0], torch.zeros(3), atol=1e-5)
    print(f"  First frame: {pos_rel[0].tolist()}")
    if first_frame_zero:
        print(f"  ✅ Relative mode: first frame is zero")
        results["relative_pose"] = True
    else:
        print(f"  ❌ Relative mode: first frame should be zero")
        results["relative_pose"] = False

    print(f"  Frame 1 relative: {pos_rel[1].tolist()}")
    print(f"  Frame 2 relative: {pos_rel[2].tolist()}")

    # Check gripper unchanged in all modes
    print("\nTesting gripper (should be unchanged):")
    gripper_abs = data_abs["action.left_gripper"]
    gripper_delta = data_delta["action.left_gripper"]
    gripper_rel = data_rel["action.left_gripper"]

    gripper_unchanged = (
        torch.allclose(gripper_abs, data["action.left_gripper"]) and
        torch.allclose(gripper_delta, data["action.left_gripper"]) and
        torch.allclose(gripper_rel, data["action.left_gripper"])
    )

    if gripper_unchanged:
        print(f"  ✅ Gripper values unchanged in all modes")
        results["gripper"] = True
    else:
        print(f"  ❌ Gripper should be unchanged")
        results["gripper"] = False

    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)

    for mode, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {mode:15s}: {status}")

    all_passed = all(results.values())
    print(f"\n  Total: {sum(results.values())}/{len(results)} passed")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = test_transform_modes()
    sys.exit(0 if success else 1)
