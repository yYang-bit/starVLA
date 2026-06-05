#!/usr/bin/env python3
"""
Lightweight dataloader test script.
Tests v2.1 and v3.0 datasets without running full training.
Only loads a few episodes to verify compatibility.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add starVLA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    ModalityConfig,
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_dataset(dataset_path: str, dataset_name: str, max_episodes: int = 3):
    """Test loading a dataset and print structure info."""

    dataset_path = Path(dataset_path)
    print_section(f"Testing: {dataset_name}")

    # Check meta files
    meta_dir = dataset_path / "meta"
    print("📁 Meta files:")
    for file in ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"]:
        path = meta_dir / file
        if path.exists():
            print(f"  ✅ {file}")
        else:
            # Check for v3.0 alternatives
            if file == "episodes.jsonl":
                alt = list(meta_dir.glob("episodes/*/*.parquet"))
                if alt:
                    print(f"  ✅ episodes/*/*.parquet (v3.0)")
                else:
                    print(f"  ❌ {file}")
            elif file == "tasks.jsonl":
                alt = meta_dir / "tasks.parquet"
                if alt.exists():
                    print(f"  ✅ tasks.parquet (v3.0)")
                else:
                    print(f"  ❌ {file}")
            else:
                print(f"  ❌ {file}")

    # Load info.json
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        print(f"\n❌ Missing info.json, cannot proceed")
        return False

    with open(info_path) as f:
        info = json.load(f)

    print(f"\n📊 Dataset info:")
    print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
    print(f"  Total frames: {info.get('total_frames', 'unknown')}")
    print(f"  FPS: {info.get('fps', 'unknown')}")

    # Check version
    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_parquet = list(meta_dir.glob("episodes/*/*.parquet"))

    if episodes_jsonl.exists():
        version = "v2.1"
        print(f"  Version: LeRobot v2.1 (episodes.jsonl)")
    elif episodes_parquet:
        version = "v3.0"
        print(f"  Version: LeRobot v3.0 (episodes parquet)")
    else:
        print(f"  ❌ Cannot detect version")
        return False

    # Print feature keys
    features = info.get("features", {})
    print(f"\n🔑 Available features ({len(features)}):")
    for key, meta in list(features.items())[:10]:  # Show first 10
        shape = meta.get("shape", "unknown")
        dtype = meta.get("dtype", "unknown")
        print(f"  - {key}: shape={shape}, dtype={dtype}")
    if len(features) > 10:
        print(f"  ... and {len(features) - 10} more")

    # Try to create a minimal dataset config
    print(f"\n🧪 Testing dataset instantiation...")

    try:
        # Detect video keys
        video_keys = [k for k in features.keys() if k.startswith("video.") or (k.startswith("observation.images") and "video" in features[k].get("dtype", ""))]

        # For Galbot, use video.* format; for FastUMI, use observation.images.*
        if not video_keys:
            # Fallback: look for any video-type features
            video_keys = [k for k in features.keys() if "camera" in k.lower() or "wrist" in k.lower() or "image" in k.lower()][:2]

        # Detect state/action keys
        state_keys = [k for k in features.keys() if k == "observation.state" or (k.startswith("state.") and "video" not in k)][:5]
        action_keys = [k for k in features.keys() if k == "action" or (k.startswith("action.") and "video" not in k)][:5]

        print(f"  Video keys: {video_keys}")
        print(f"  State keys: {state_keys[:3]}{'...' if len(state_keys) > 3 else ''}")
        print(f"  Action keys: {action_keys[:3]}{'...' if len(action_keys) > 3 else ''}")

        # Create minimal config
        modality_configs = {}

        if video_keys:
            modality_configs["video"] = ModalityConfig(
                delta_indices=[0],
                modality_keys=video_keys,
            )

        if state_keys:
            modality_configs["state"] = ModalityConfig(
                delta_indices=[0],
                modality_keys=state_keys,
            )

        if action_keys:
            modality_configs["action"] = ModalityConfig(
                delta_indices=list(range(0, 30, 2)),  # Sample every 2 frames for 15 actions
                modality_keys=action_keys,
            )

        # Create dataset (don't load transforms yet)
        from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # Use generic tag for testing
            data_cfg={"lerobot_version": "auto"},  # Auto-detect
        )

        print(f"  ✅ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")

        # Load first sample
        print(f"\n🎯 Loading first sample...")
        sample = dataset[0]

        print(f"  Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape} {value.dtype}")
            elif isinstance(value, list):
                print(f"    {key}: list[{len(value)}]")
            else:
                print(f"    {key}: {type(value)}")

        # Try loading a few more samples
        print(f"\n📦 Loading {min(3, len(dataset))} samples...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"  Sample {i}: action shape = {sample.get('action', torch.tensor([])).shape}")

        print(f"\n✅ Dataset test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Dataset test FAILED:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Lightweight dataloader test")
    parser.add_argument("--v21-fastumi", type=str,
                       default="/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker",
                       help="Path to FastUMI v2.1 dataset")
    parser.add_argument("--v30-galbot", type=str,
                       default="/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0529_359piece",
                       help="Path to Galbot v3.0 dataset")
    parser.add_argument("--skip-v21", action="store_true", help="Skip v2.1 test")
    parser.add_argument("--skip-v30", action="store_true", help="Skip v3.0 test")

    args = parser.parse_args()

    results = {}

    # Test v2.1 (FastUMI)
    if not args.skip_v21:
        results["v2.1_fastumi"] = test_dataset(
            args.v21_fastumi,
            "FastUMI (LeRobot v2.1)",
            max_episodes=3
        )

    # Test v3.0 (Galbot)
    if not args.skip_v30:
        results["v3.0_galbot"] = test_dataset(
            args.v30_galbot,
            "Galbot (LeRobot v3.0)",
            max_episodes=3
        )

    # Summary
    print_section("Test Summary")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
