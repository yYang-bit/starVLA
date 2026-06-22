# SPDX-FileCopyrightText: Copyright (c) 2025 STARVLA Authors
# SPDX-License-Identifier: Apache-2.0

"""
BinaryGripperTransform — 本任务专用（不进框架）

将 gripper 真值二值化为 {0, 1}，绕过 StateActionTransform 的归一化。

设计动机:
    gripper 二值化只需一个阈值，不需要 stats。但 StateActionTransform 的 binary
    模式强制要求 stats 校验（且必须是 [0] 或 [1]），与"大规模预训练用同一套真值
    stats"的目标冲突。因此本任务采用独立 transform 绕过归一化体系:
      1. gripper 在 StateActionTransform 中不归一化（保持真值，mm）
      2. 本 transform 在 normalization 之后做 (x > threshold) → {0, 1}
    这样 stats.json 仍按真值统计 gripper 分布，可供大规模预训练复用。

放置位置:
    必须在 StateActionTransform 之后（否则真值会被 q99 改写，阈值失效）。

用法:
    BinaryGripperTransform(
        apply_to=["action.left_gripper", "action.right_gripper"],
        threshold=100.0,  # mm
    )
"""

from typing import Any

import numpy as np
import torch
from pydantic import Field

from starVLA.dataloader.gr00t_lerobot.transform.base import ModalityTransform


class BinaryGripperTransform(ModalityTransform):
    """Threshold-based binarization for gripper keys.

    Maps gripper raw values to {0, 1} via (x > threshold), without any stats.
    """

    threshold: float = Field(
        default=100.0,
        description="Threshold for binarization (raw gripper units, e.g. mm). "
                    "Values > threshold → 1, else → 0.",
    )

    def _ensure_tensor(self, data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        return torch.from_numpy(np.asarray(data, dtype=np.float32))

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue

            x = self._ensure_tensor(data[key])
            original_dtype = x.dtype

            # (x > threshold) → {0, 1}
            binary = (x > self.threshold).to(original_dtype)

            data[key] = binary

        return data
