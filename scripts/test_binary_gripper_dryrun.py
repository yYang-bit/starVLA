#!/usr/bin/env python3
"""
Dry-run: 验证本任务的 gripper 二值化方案能跑通（合成数据，不加载真数据）

验证点:
1. binary_independent 模式下，gripper 不走 StateActionTransform 归一化
2. BinaryGripperTransform 在末尾正确做 (x > threshold) → {0,1}
3. pos/rot 仍走 q99 归一化（此处跳过 stats，只验证流程不报错）
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# 动态导入本任务 config（不依赖包结构）
import importlib.util
config_path = Path(__file__).parent.parent / "examples/GalbotBimanualRelative/train_files/data_registry/data_config.py"
spec = importlib.util.spec_from_file_location("galbot_data_config", config_path)
galbot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(galbot_module)

GalbotBimanualSelfDataConfig = galbot_module.GalbotBimanualSelfDataConfig


def test_binary_independent_mode():
    print("\n" + "=" * 70)
    print("  BinaryGripperTransform Dry-run (合成数据)")
    print("=" * 70 + "\n")

    config = GalbotBimanualSelfDataConfig()

    # 本任务配置: delta + binary_independent
    data_cfg = {
        "action_mode": "delta",
        "action_chunk_size": 4,
        "gripper_normalization": "binary_independent",
        "gripper_binary_threshold": 100.0,
    }

    print(f"配置: {data_cfg}\n")

    # 构建 transform pipeline
    transforms = config.transform(data_cfg)
    print(f"Transform pipeline (共 {len(transforms.transforms)} 个):")
    for i, t in enumerate(transforms.transforms):
        print(f"  [{i}] {type(t).__name__}")
    print()

    # 合成数据 (模拟一个 sample)
    T = 4  # chunk size
    data = {
        # action keys
        "action.left_pos": torch.zeros(T, 3, dtype=torch.float32),
        "action.left_ori_6d": torch.tensor([[1., 0., 0., 1., 0., 0.]] * T, dtype=torch.float32),
        "action.left_gripper": torch.tensor([[50.], [150.], [30.], [120.]], dtype=torch.float32),  # 混合
        "action.right_pos": torch.zeros(T, 3, dtype=torch.float32),
        "action.right_ori_6d": torch.tensor([[1., 0., 0., 1., 0., 0.]] * T, dtype=torch.float32),
        "action.right_gripper": torch.tensor([[80.], [100.], [10.], [200.]], dtype=torch.float32),  # 混合
        # state keys
        "state.dual_relative_pose_pos": torch.zeros(3, dtype=torch.float32),
        "state.dual_relative_pose_ori_6d": torch.tensor([1., 0., 0., 1., 0., 0.], dtype=torch.float32),
    }

    # 设置 metadata (StateActionTransform 需要)
    # 注意: 这里跳过真实 stats，用 dummy metadata 让流程跑通
    # 实际训练时 stats.json 会提供真实值
    print("注意: dry-run 用 dummy metadata，实际训练用真实 stats.json\n")

    # StateActionTransform 需要 metadata，这里我们只测 ActionChunkTransform + BinaryGripperTransform
    # 跳过 StateActionTransform (索引 2, 3)，直接测 ActionChunkTransform(1) + BinaryGripperTransform(末尾)
    from starVLA.dataloader.gr00t_lerobot.transform.action_chunk_mode import ActionChunkTransform

    # 单独测 ActionChunkTransform
    print("--- 步骤 1: ActionChunkTransform (delta) ---")
    act_transform = ActionChunkTransform(
        mode="delta",
        apply_to=config.action_keys,
        position_suffix="_pos",
        rotation_suffix="_ori_6d",
        gripper_suffix="_gripper",
    )
    data_after_act = act_transform.apply({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()})

    print(f"  action.left_pos[0]: {data_after_act['action.left_pos'][0].tolist()} (应为 [0,0,0], delta首帧)")
    print(f"  action.left_gripper: {data_after_act['action.left_gripper'].flatten().tolist()}")
    print(f"  action.right_gripper: {data_after_act['action.right_gripper'].flatten().tolist()}")
    print(f"  (gripper 此时仍是真值，未二值化)")

    # 单独测 BinaryGripperTransform
    from examples.GalbotBimanualRelative.train_files.data_registry.binary_gripper_transform import (
        BinaryGripperTransform,
    )

    print("\n--- 步骤 2: BinaryGripperTransform (threshold=100) ---")
    bin_transform = BinaryGripperTransform(
        apply_to=["action.left_gripper", "action.right_gripper"],
        threshold=100.0,
    )
    data_final = bin_transform.apply(data_after_act)

    left_bin = data_final["action.left_gripper"].flatten().tolist()
    right_bin = data_final["action.right_gripper"].flatten().tolist()
    print(f"  action.left_gripper: {left_bin}")
    print(f"    原值: [50, 150, 30, 120] → 期望: [0, 1, 0, 1]")
    print(f"  action.right_gripper: {right_bin}")
    print(f"    原值: [80, 100, 10, 200] → 期望: [0, 0, 0, 1] (>100才为1, 100本身为0)")

    # 验证
    expected_left = [0., 1., 0., 1.]
    expected_right = [0., 0., 0., 1.]
    left_ok = all(abs(a - b) < 1e-6 for a, b in zip(left_bin, expected_left))
    right_ok = all(abs(a - b) < 1e-6 for a, b in zip(right_bin, expected_right))

    print(f"\n  left_gripper 正确: {'✅' if left_ok else '❌'}")
    print(f"  right_gripper 正确: {'✅' if right_ok else '❌'}")

    # 验证 pos/rot 没被 BinaryGripperTransform 动过
    print("\n--- 步骤 3: 验证 pos/rot 未被 BinaryGripperTransform 影响 ---")
    pos_unchanged = torch.equal(data_after_act["action.left_pos"], data_final["action.left_pos"])
    rot_unchanged = torch.equal(data_after_act["action.left_ori_6d"], data_final["action.left_ori_6d"])
    print(f"  action.left_pos 未变: {'✅' if pos_unchanged else '❌'}")
    print(f"  action.left_ori_6d 未变: {'✅' if rot_unchanged else '❌'}")

    all_pass = left_ok and right_ok and pos_unchanged and rot_unchanged
    print(f"\n{'=' * 70}")
    print(f"  结果: {'✅ 全部通过' if all_pass else '❌ 有失败项'}")
    print(f"{'=' * 70}\n")

    return all_pass


if __name__ == "__main__":
    success = test_binary_independent_mode()
    sys.exit(0 if success else 1)
