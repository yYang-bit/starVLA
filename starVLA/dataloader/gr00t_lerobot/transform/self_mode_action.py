# SPDX-FileCopyrightText: Copyright (c) 2025 STARVLA Authors
# SPDX-License-Identifier: Apache-2.0

"""
SelfModeActionTransform — 将 raw abs action 转为训练用的 self-mode action。

三种模式：
    "abs"             : 不变，直接通过
    "delta"           : 逐帧差分（t>0: a[t] - a[t-1], t=0: 0）
    "chunk_relative"  : chunk 内以第 0 帧为 base（a[t] - a[0]）

不依赖外部 state 列，所有计算都在 action chunk 内部完成。

放置位置：StateActionToTensor 之后、StateActionTransform 之间。
"""

from typing import Any

from .base import ModalityTransform


class SelfModeActionTransform(ModalityTransform):
    """
    Transform raw abs action into self-mode action.

    Args:
        apply_to: List of action keys to transform, e.g. ["action.left_pos", ...]
        self_mode: "abs" | "delta" | "chunk_relative"
    """

    self_mode: str = "abs"

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.self_mode == "abs":
            return data

        for key in self.apply_to:
            if key not in data:
                continue

            x = data[key]  # (chunk_size, dim), torch tensor

            if self.self_mode == "chunk_relative":
                data[key] = x - x[0:1]

            elif self.self_mode == "delta":
                out = x.clone()
                if out.shape[0] > 1:
                    out[1:] = x[1:] - x[:-1]
                out[0:1] = 0  # 第一帧的 delta 为 0
                data[key] = out

        return data