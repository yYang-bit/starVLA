#!/usr/bin/env python3
"""
轻量化数据加载测试 - 只测试能否加载数据和 transform 是否工作
不限制 episodes（因为 LeRobotSingleDataset 可能不支持），但只读取前 3 个样本
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

# 动态导入 FastUMI 配置
import importlib.util
config_path = Path(__file__).parent.parent / "examples/FastUMI/train_files/data_registry/data_config.py"
spec = importlib.util.spec_from_file_location("fastumi_config", config_path)
fastumi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fastumi_module)

FastUMIDualArmDataConfig = fastumi_module.FastUMIDualArmDataConfig


def test_fastumi_lightweight():
    """轻量化测试：只加载前 3 个样本"""

    print("\n" + "="*70)
    print("  FastUMI 轻量化数据加载测试（仅前 3 个样本）")
    print("="*70 + "\n")

    dataset_path = Path("/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker")

    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False

    config = FastUMIDualArmDataConfig()
    modes = ["abs", "delta", "relative_pose"]

    results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  测试模式: {mode}")
        print(f"{'='*70}\n")

        # 构建 data_cfg
        data_cfg = {
            "action_mode": mode,
            "action_chunk_size": 4,  # 小 chunk，快速测试
            "gripper_normalization": None,
        }

        print(f"配置: {data_cfg}")

        # 创建数据集（使用 transform_for_stats，无归一化）
        try:
            dataset = LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=config.modality_config(data_cfg),
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                transforms=config.transform_for_stats(data_cfg),
            )
            print(f"✅ 数据集创建成功，总样本数: {len(dataset)}")
        except Exception as e:
            print(f"❌ 数据集创建失败: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = False
            continue

        # 只测试前 3 个样本
        print(f"\n开始测试前 3 个样本...")
        success_count = 0

        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]

                # 检查关键 action keys
                action_pos_key = "action.left_arm"
                action_rot_key = "action.left_ori_6d"

                if action_pos_key not in sample:
                    print(f"  ❌ 样本 {i}: 缺少 {action_pos_key}")
                    continue

                pos = sample[action_pos_key]
                rot = sample[action_rot_key]

                print(f"  ✅ 样本 {i}: pos shape={pos.shape}, rot shape={rot.shape}")

                # 验证首帧行为
                if i == 0:  # 只在第一个样本验证
                    pos_0 = pos[0].numpy() if hasattr(pos, 'numpy') else pos[0]
                    import numpy as np
                    max_val = np.abs(pos_0).max()

                    if mode == "delta":
                        if max_val < 1e-5:
                            print(f"    ✓ Delta 模式: 首帧为 0 (max={max_val:.6f})")
                        else:
                            print(f"    ⚠️  Delta 模式: 首帧不为 0 (max={max_val:.6f})")

                    elif mode == "relative_pose":
                        if max_val < 1e-5:
                            print(f"    ✓ Relative 模式: 首帧为 0 (max={max_val:.6f})")
                        else:
                            print(f"    ⚠️  Relative 模式: 首帧不为 0 (max={max_val:.6f})")

                success_count += 1

            except Exception as e:
                print(f"  ❌ 样本 {i} 加载失败: {e}")
                import traceback
                traceback.print_exc()

        if success_count == 3:
            print(f"\n✅ 模式 '{mode}' 测试通过 (3/3 样本成功)")
            results[mode] = True
        else:
            print(f"\n⚠️  模式 '{mode}' 部分成功 ({success_count}/3 样本)")
            results[mode] = False

    # 总结
    print(f"\n{'='*70}")
    print(f"  测试总结")
    print(f"{'='*70}\n")

    for mode in modes:
        status = "✅ PASSED" if results.get(mode, False) else "❌ FAILED"
        print(f"  {mode:15s}: {status}")

    success_count = sum(results.values())
    total_count = len(modes)
    print(f"\n  通过: {success_count}/{total_count}")
    print(f"{'='*70}\n")

    return success_count == total_count


if __name__ == "__main__":
    import sys
    success = test_fastumi_lightweight()
    sys.exit(0 if success else 1)
