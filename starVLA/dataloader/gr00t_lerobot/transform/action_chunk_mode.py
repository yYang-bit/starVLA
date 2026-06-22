# SPDX-FileCopyrightText: Copyright (c) 2025 STARVLA Authors
# SPDX-License-Identifier: Apache-2.0

"""
ActionChunkTransform - Unified action representation for embodiment-agnostic learning.

Supports three modes for action chunk transformation:
1. "abs": Absolute pose in world frame (no transform, pass-through)
2. "delta": Frame-to-frame delta in local end-effector frame
3. "relative_pose": Relative to base frame (action[0]) in local frame

All modes work purely on action chunk - no state dependency.
State is only used as model input, not for data transformation.

Key formulas (local/body-frame representation):
    delta mode:
        delta_xyz[t] = R[t-1].T @ (p[t] - p[t-1])
        delta_R[t]   = R[t-1].T @ R[t]

    relative_pose mode:
        rel_xyz[t] = R[0].T @ (p[t] - p[0])
        rel_R[t]   = R[0].T @ R[t]

This local representation supports cross-embodiment learning by encoding
task-level motion patterns rather than absolute positions.

Transform pipeline:
    StateActionToTensor → ActionChunkTransform → StateActionTransform (normalize)

Usage:
    ActionChunkTransform(
        mode="relative_pose",
        action_keys=["action.left_arm", "action.left_ori_6d", ...],
        position_suffix="_arm",
        rotation_suffix="_ori_6d",
        gripper_suffix="_gripper",
    )
"""

from typing import Any

import torch
import numpy as np
from pydantic import Field, model_validator

from .base import ModalityTransform
from .rotation_utils import rot6d_to_matrix, matrix_to_rot6d


class ActionChunkTransform(ModalityTransform):
    """
    Unified action chunk transform for embodiment-agnostic representations.

    Works purely on action chunk [T, D] - no state dependency.
    Gripper values are preserved (not differenced).

    Args:
        mode: Transform mode - "abs" | "delta" | "relative_pose"
        action_keys: Action keys to transform (e.g., ["action.left_arm", ...])
        position_suffix: Suffix to identify position keys (default: "_arm")
        rotation_suffix: Suffix to identify rotation keys (default: "_ori_6d")
        gripper_suffix: Suffix to identify gripper keys (default: "_gripper")
    """

    mode: str = Field(
        ...,
        description="Transform mode: abs (no transform) | delta (frame-to-frame) | relative_pose (relative to base)"
    )
    apply_to: list[str] = Field(
        ...,
        description="Action keys to transform (position, rotation, gripper)"
    )
    position_suffix: str = Field(
        default="_arm",
        description="Suffix to identify position keys"
    )
    rotation_suffix: str = Field(
        default="_ori_6d",
        description="Suffix to identify rotation keys (rotation_6d format)"
    )
    gripper_suffix: str = Field(
        default="_gripper",
        description="Suffix to identify gripper keys (preserved as absolute)"
    )

    @model_validator(mode='after')
    def validate_mode(self):
        """Validate mode parameter."""
        valid_modes = {"abs", "delta", "relative_pose"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got {self.mode!r}"
            )
        return self

    def _is_position_key(self, key: str) -> bool:
        """Check if key is a position key (e.g., action.left_arm)."""
        return self.position_suffix in key and self.gripper_suffix not in key

    def _is_rotation_key(self, key: str) -> bool:
        """Check if key is a rotation key (e.g., action.left_ori_6d)."""
        return self.rotation_suffix in key

    def _is_gripper_key(self, key: str) -> bool:
        """Check if key is a gripper key (e.g., action.left_gripper)."""
        return self.gripper_suffix in key

    def _ensure_tensor(self, data: Any) -> torch.Tensor:
        """Convert data to torch.Tensor if needed."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.from_numpy(np.asarray(data, dtype=np.float32))

    def _delta_position(
        self,
        pos_chunk: torch.Tensor,  # [T, 3]
        rot_chunk: torch.Tensor,  # [T, 6] rotation_6d
    ) -> torch.Tensor:
        """
        Frame-to-frame position delta in local coordinate frame.

        Formula: delta_xyz[t] = R[t-1].T @ (p[t] - p[t-1])

        This encodes local motion pattern (e.g., "move forward 10cm in body frame")
        which is embodiment-agnostic.

        Args:
            pos_chunk: Position chunk [T, 3]
            rot_chunk: Rotation chunk [T, 6] in rotation_6d format

        Returns:
            delta_chunk: Delta positions [T, 3] in local frames
        """
        T = pos_chunk.shape[0]
        delta = torch.zeros_like(pos_chunk)

        # Convert rotation_6d to rotation matrix
        rot_mat = rot6d_to_matrix(rot_chunk)  # [T, 3, 3]
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            rot_mat = torch.from_numpy(rot_mat).to(dtype=pos_chunk.dtype, device=pos_chunk.device)

        # Compute frame-to-frame delta
        for t in range(T - 1, 0, -1):
            # World frame delta
            delta_world = pos_chunk[t] - pos_chunk[t - 1]

            # Transform to previous frame's local coordinate system
            R_prev_inv = rot_mat[t - 1].T  # [3, 3]
            delta[t] = R_prev_inv @ delta_world

        # First frame: set to zero (no previous frame)
        delta[0] = torch.zeros(3, dtype=pos_chunk.dtype, device=pos_chunk.device)

        return delta

    def _relative_position(
        self,
        pos_chunk: torch.Tensor,  # [T, 3]
        rot_chunk: torch.Tensor,  # [T, 6] rotation_6d
    ) -> torch.Tensor:
        """
        Position relative to base frame (action[0]) in local coordinates.

        Formula: rel_xyz[t] = R[0].T @ (p[t] - p[0])

        This encodes task-level motion pattern relative to initial pose,
        supporting cross-embodiment transfer.

        Args:
            pos_chunk: Position chunk [T, 3]
            rot_chunk: Rotation chunk [T, 6] in rotation_6d format

        Returns:
            rel_chunk: Relative positions [T, 3] in base local frame
        """
        # Base frame (action[0])
        base_pos = pos_chunk[0]  # [3]
        base_rot = rot6d_to_matrix(rot_chunk[0])  # [3, 3]
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            base_rot = torch.from_numpy(base_rot).to(dtype=pos_chunk.dtype, device=pos_chunk.device)
        base_rot_inv = base_rot.T

        # World frame offset
        offset_world = pos_chunk - base_pos  # [T, 3]

        # Transform to base local coordinate system
        rel = (base_rot_inv @ offset_world.T).T  # [T, 3]

        return rel

    def _delta_rotation(
        self,
        rot_chunk: torch.Tensor,  # [T, 6] rotation_6d
    ) -> torch.Tensor:
        """
        Frame-to-frame rotation delta in SO(3).

        Formula: delta_R[t] = R[t-1].T @ R[t]

        This is NOT arithmetic subtraction! Rotation difference must use
        SO(3) group operation to preserve rotation constraints.

        Args:
            rot_chunk: Rotation chunk [T, 6] in rotation_6d format

        Returns:
            delta_rot: Delta rotations [T, 6] in rotation_6d format
        """
        # Convert to rotation matrix
        rot_mat = rot6d_to_matrix(rot_chunk)  # [T, 3, 3]
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            rot_mat = torch.from_numpy(rot_mat).to(dtype=rot_chunk.dtype, device=rot_chunk.device)
        T = rot_mat.shape[0]

        delta_mat = torch.zeros_like(rot_mat)

        # Compute frame-to-frame relative rotation
        for t in range(T - 1, 0, -1):
            # SO(3) relative rotation
            R_prev_inv = rot_mat[t - 1].T
            delta_mat[t] = R_prev_inv @ rot_mat[t]

        # First frame: identity rotation (no rotation change)
        delta_mat[0] = torch.eye(
            3,
            dtype=rot_mat.dtype,
            device=rot_mat.device
        )

        # Convert back to rotation_6d
        result = matrix_to_rot6d(delta_mat)
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            result = torch.from_numpy(result).to(dtype=rot_chunk.dtype, device=rot_chunk.device)
        return result

    def _relative_rotation(
        self,
        rot_chunk: torch.Tensor,  # [T, 6] rotation_6d
    ) -> torch.Tensor:
        """
        Rotation relative to base frame (action[0]) in SO(3).

        Formula: rel_R[t] = R[0].T @ R[t]

        All frames expressed relative to initial orientation.

        Args:
            rot_chunk: Rotation chunk [T, 6] in rotation_6d format

        Returns:
            rel_rot: Relative rotations [T, 6] in rotation_6d format
        """
        # Convert to rotation matrix
        rot_mat = rot6d_to_matrix(rot_chunk)  # [T, 3, 3]
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            rot_mat = torch.from_numpy(rot_mat).to(dtype=rot_chunk.dtype, device=rot_chunk.device)

        # Base frame rotation (action[0])
        base_rot = rot_mat[0]  # [3, 3]
        base_rot_inv = base_rot.T

        # All frames relative to base
        # Using einsum for efficient batched matrix multiplication
        rel_mat = torch.einsum("ij,tjk->tik", base_rot_inv, rot_mat)

        # Convert back to rotation_6d
        result = matrix_to_rot6d(rel_mat)
        # Convert to tensor if input was tensor
        if isinstance(rot_chunk, torch.Tensor):
            result = torch.from_numpy(result).to(dtype=rot_chunk.dtype, device=rot_chunk.device)
        return result

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply action chunk transform.

        Processes position and rotation keys according to mode.
        Gripper keys are preserved as absolute values (not differenced).

        Args:
            data: Sample dict with action keys [T, D]

        Returns:
            data: Transformed sample dict
        """
        if self.mode == "abs":
            # Absolute mode: no transform, pass through
            return data

        # Process each action key
        for action_key in self.apply_to:
            if action_key not in data:
                continue

            # Gripper: preserve absolute values (no transform)
            if self._is_gripper_key(action_key):
                continue

            # Get action chunk
            action_chunk = data[action_key]
            action_chunk = self._ensure_tensor(action_chunk)

            # Save original dtype and device
            original_dtype = action_chunk.dtype
            original_device = action_chunk.device

            # Transform based on key type
            if self._is_position_key(action_key):
                # Position: need corresponding rotation for coordinate transform
                rot_key = action_key.replace(
                    self.position_suffix,
                    self.rotation_suffix
                )

                if rot_key not in data:
                    raise KeyError(
                        f"Rotation key {rot_key} not found for position key {action_key}. "
                        f"Position transform requires rotation for coordinate frame conversion."
                    )

                rot_chunk = self._ensure_tensor(data[rot_key])

                # Apply position transform
                if self.mode == "delta":
                    transformed = self._delta_position(action_chunk, rot_chunk)
                else:  # relative_pose
                    transformed = self._relative_position(action_chunk, rot_chunk)

                # Restore original dtype/device
                data[action_key] = transformed.to(
                    dtype=original_dtype,
                    device=original_device
                )

            elif self._is_rotation_key(action_key):
                # Rotation: SO(3) operation
                if self.mode == "delta":
                    transformed = self._delta_rotation(action_chunk)
                else:  # relative_pose
                    transformed = self._relative_rotation(action_chunk)

                # Restore original dtype/device
                data[action_key] = transformed.to(
                    dtype=original_dtype,
                    device=original_device
                )

        return data
