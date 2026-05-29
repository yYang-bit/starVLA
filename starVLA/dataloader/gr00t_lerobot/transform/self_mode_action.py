# SPDX-FileCopyrightText: Copyright (c) 2025 STARVLA Authors
# SPDX-License-Identifier: Apache-2.0

"""
SelfModeActionTransform — 将 raw abs action 转为训练用的 self-mode action。

三种模式：
    "abs"             : 不变，直接通过
    "delta"           : 逐帧差分
                        - pos: 算术差分 (a[t] - a[t-1], a[0] = 0)
                        - rotation_6d: SO(3) 相对旋转 (R[t] @ R[t-1]^T, R[0] = I)
    "chunk_relative"  : chunk 内以第 0 帧为 base
                        - pos: a[t] - a[0]
                        - rotation_6d: R[t] @ R[0]^T

不依赖外部 state 列，所有计算都在 action chunk 内部完成。

放置位置：StateActionToTensor 之后、StateActionTransform 之间。
"""

from typing import Any

import torch

from .base import ModalityTransform
from .rotation_utils import (
    compute_relative_rotation_rot6d,
    identity_rotation_rot6d,
    rot6d_to_matrix,
    matrix_to_rot6d,
)


class SelfModeActionTransform(ModalityTransform):
    """
    Transform raw abs action into self-mode action.

    Args:
        apply_to: List of action keys to transform, e.g. ["action.left_pos", ...]
        self_mode: "abs" | "delta" | "chunk_relative"
        rotation_keys: List of keys that represent rotation_6d (require SO(3) operations)
    """

    self_mode: str = "abs"
    rotation_keys: list[str] = []

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.self_mode == "abs":
            return data

        rotation_keys_set = set(self.rotation_keys)

        for key in self.apply_to:
            if key not in data:
                continue

            x = data[key]  # (chunk_size, dim), torch tensor
            is_rotation = key in rotation_keys_set

            if is_rotation:
                # SO(3) operations on rotation_6d
                x_np = x.cpu().numpy()  # (T, 6)

                if self.self_mode == "chunk_relative":
                    # R_rel[t] = R[t] @ R[0]^T
                    out_np = compute_relative_rotation_rot6d(x_np, x_np[0:1])

                elif self.self_mode == "delta":
                    # R_delta[t] = R[t] @ R[t-1]^T, R_delta[0] = I
                    out_np = x_np.copy()
                    if out_np.shape[0] > 1:
                        for t in range(out_np.shape[0] - 1, 0, -1):
                            out_np[t] = compute_relative_rotation_rot6d(
                                x_np[t], x_np[t - 1]
                            )
                    out_np[0] = identity_rotation_rot6d()

                data[key] = torch.from_numpy(out_np).to(x.device, dtype=x.dtype)

            else:
                # Arithmetic operations on position/other continuous dims
                if self.self_mode == "chunk_relative":
                    data[key] = x - x[0:1]

                elif self.self_mode == "delta":
                    out = x.clone()
                    if out.shape[0] > 1:
                        out[1:] = x[1:] - x[:-1]
                    out[0:1] = 0
                    data[key] = out

        return data