#!/usr/bin/env python3
"""
超轻量化数据加载测试 - 只读取 2-3 个 episodes
测试 ActionChunkTransform 的三种模式
"""

import sys
import torch
from pathlib import Path

# Add starVLA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.action_chunk_mode import ActionChunkTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

# 导入 FastUMI 配置
fastumi_config_path = Path("/mnt/home/liuyi/project/starVLA/examples/FastUMI/train_files/data_registry/data_config.py")
sys.path.insert(0, str(fastumi_config_path.parent.parent.parent))
from data_registry.data_config import FastUMIDataConfig


def test_fastumi_data_loading():
    """
    测试 FastUMI 数据加载 + ActionChunkTransform 三种模式
    只加载 2 个 episodes，确保轻量化
    """

    print("\n" + "="*70)
    print("FastUMI 轻量化数据加载测试（仅 2 个 episodes）")
    print("="*70 + "\n")

    # 数据集路径
    dataset_path = "/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker"

    # 配置
    config = FastUMIDataConfig()
    action_keys = config.action_keys
    state_keys = config.state_keys

    # 三种模式
    modes = ["abs", "delta", "relative_pose"]

    for mode in modes:
        print(f"\n--- 测试模式: {mode} ---")

        # 创建数据集（只加载 2 个 episodes）
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config.modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            episodes=[0, 1],  # 只加载 episode 0 和 1
            action_chunk_transform=ActionChunkTransform(
                mode=mode,
                apply_to=action_keys,
                state_keys=state_keys
            )
        )

        print(f"✓ 数据集加载成功: {len(dataset)} samples (from 2 episodes)")

        # 测试第一个样本
        sample = dataset[0]

        # 检查 action keys
        print(f"\nAction keys 形状:")
        for key in action_keys:
            if key in sample:
                shape = sample[key].shape
                dtype = sample[key].dtype
                print(f"  {key}: {shape} ({dtype})")

                # 验证模式特性
                if mode == "delta":
                    # Delta 模式：第一帧应该是 0
                    first_frame = sample[key][0]
                    if "pos" in key or "arm" in key:
                        is_zero = torch.allclose(first_frame, torch.zeros_like(first_frame), atol=1e-6)
                        print(f"    ✓ 第一帧为 0: {is_zero}")

                elif mode == "relative_pose":
                    # Relative 模式：第一帧应该是 0（相对于自身）
                    first_frame = sample[key][0]
                    if "pos" in key or "arm" in key:
                        is_zero = torch.allclose(first_frame, torch.zeros_like(first_frame), atol=1e-6)
                        print(f"    ✓ 第一帧为 0: {is_zero}")

        # 检查 state keys
        print(f"\nState keys 形状:")
        for key in state_keys:
            if key in sample:
                shape = sample[key].shape
                dtype = sample[key].dtype
                print(f"  {key}: {shape} ({dtype})")

        print(f"\n✓ 模式 {mode} 测试通过")

    print("\n" + "="*70)
    print("✓ 所有模式测试完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_fastumi_data_loading()
