"""FastUMI dual-arm dataset configuration with unified action chunk transform.

This config demonstrates:
1. Flat vector format (observation.state: [14D], action: [14D])
2. DerivedKeysTransform to split vectors into structured keys
3. RPY → rotation_6d conversion
4. ActionChunkTransform for embodiment-agnostic action representation

Action transform modes (controlled by data_cfg.action_mode):
    "abs"             : Absolute pose in world frame (no transform)
    "delta"           : Frame-to-frame delta in local end-effector frame
    "relative_pose"   : Relative to base (action[0]) in local frame

Configuration (all optional, defaults shown):
    action_mode: "abs"                     # Transform mode
    action_chunk_size: 16                  # Action chunk horizon (sampled at 2-step interval)
    gripper_normalization: None            # None (use raw values) | "binary" | "min_max"
    gripper_binary_threshold: 0.5          # Threshold for binary mode
"""

from typing import Any

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.action_chunk_mode import (
    ActionChunkTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


# FastUMI task list (for reference)
FASTUMI_DUAL_ARM_TASKS = [
    "Add_Rice_to_Rice_Cooker",
    "Arrange_Toothbrush_and_Toothpaste",
    "Clean_Desktop",
    "Dispose_of_Desktop_Debris",
    "Fold_the_Jeans",
    "Fold_the_Suit",
    "Fold_the_T-shirt",
    "Open_Double_Door_Cabinet",
]


class DerivedKeysTransform(ModalityTransform):
    """Transform to derive structured keys from flat vectors.

    Supports:
    1. Slicing: Extract sub-vectors (keep_from_origin)
    2. Rotation conversion: RPY/Quaternion → rotation_6d (transform_from_origin)

    Example:
        observation.state: [14D] flat vector
        ↓
        state.left_arm: [3D] (slice [0:3])
        state.left_ori_6d: [6D] (RPY→rot6d on [3:6])
        state.left_gripper: [1D] (slice [6:7])
    """

    derived_keys: dict[str, dict[str, Any]] = Field(
        ..., description="Mapping from output key to source slicing or representation transform spec."
    )
    drop_source_keys: bool = Field(default=False, description="Remove raw keys after all derived outputs are created.")

    def _slice_source(self, source: torch.Tensor | np.ndarray, start: int, end: int):
        """Slice a sub-vector from source."""
        return source[..., start:end]

    def _rotation_to_6d(self, value, spec: dict):
        """Convert rotation to rotation_6d format."""
        source_repr = str(spec.get("from", "")).lower()

        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(np.asarray(value, dtype=np.float32))
        value = value.to(torch.float32)

        if source_repr in {"rpy", "euler", "euler_angles", "euler_xyz"}:
            matrix = pt.euler_angles_to_matrix(value, convention="XYZ")
        elif source_repr in {"quaternion", "quat"}:
            matrix = pt.quaternion_to_matrix(value)
        elif source_repr in {"rotation_6d", "rot6d"}:
            return value
        else:
            raise ValueError(f"Unsupported rotation source: {source_repr}")

        return pt.matrix_to_rotation_6d(matrix)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply derived keys transform."""
        used_source_keys: set[str] = set()

        for output_key, spec in self.derived_keys.items():
            source_key = str(spec.get("source_key", ""))
            if source_key not in data:
                continue

            source = data[source_key]
            source_is_tensor = isinstance(source, torch.Tensor)
            source_device = source.device if source_is_tensor else None
            source_dtype = source.dtype if source_is_tensor else None

            if not source_is_tensor:
                source = torch.from_numpy(np.asarray(source, dtype=np.float32))

            source = source.to(torch.float32)
            start = int(spec.get("start", 0))
            end = int(spec.get("end", source.shape[-1]))
            source_slice = source[..., start:end]

            spec_type = str(spec.get("type", "")).lower()

            if spec_type in {"keep_from_origin", "keep_from_orign"}:
                output = source_slice
            elif spec_type in {"transform_from_origin", "transform_from_orign"}:
                output = self._rotation_to_6d(source_slice, spec)
            else:
                raise ValueError(f"Unsupported derived key type for {output_key}: {spec.get('type')}")

            # Restore original type
            if source_is_tensor:
                data[output_key] = output.to(device=source_device, dtype=source_dtype)
            else:
                data[output_key] = output.detach().cpu().numpy().astype(np.float32)

            used_source_keys.add(source_key)

        if self.drop_source_keys:
            for source_key in used_source_keys:
                data.pop(source_key, None)

        return data


class FastUMIDualArmDataConfig:
    """FastUMI dual-arm robot configuration.

    Data format:
    - observation.state: [14D] = [left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
    - action: [14D] = same structure

    Transform pipeline:
    1. DerivedKeysTransform: Split flat vectors into structured keys + RPY→rot6d
    2. ActionChunkTransform: Transform to local frame (delta or relative_pose)
    3. StateActionTransform: Normalize
    """

    # Default configuration
    default_action_mode = "abs"
    default_action_chunk_size = 16  # 16 actions sampled at 2-step interval
    default_gripper_normalization = None
    default_gripper_binary_threshold = 0.5

    video_keys = [
        "video.observation.images.left_camera_rgb_image",
        "video.observation.images.right_camera_rgb_image",
    ]

    # Raw keys from parquet (flat vectors)
    raw_state_keys = ["observation.state"]  # [14D]
    raw_action_keys = ["action"]            # [14D]

    # Derived keys after DerivedKeysTransform
    state_keys = [
        "state.left_arm",
        "state.left_ori_6d",
        "state.left_gripper",
        "state.right_arm",
        "state.right_ori_6d",
        "state.right_gripper",
    ]

    action_keys = [
        "action.left_arm",
        "action.left_ori_6d",
        "action.left_gripper",
        "action.right_arm",
        "action.right_ori_6d",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    # Sampling indices (fixed for observation/state, dynamic for action)
    observation_indices = [0]
    state_indices = [0]
    # action_indices is dynamic (generated in modality_config)

    # Derivation rules for DerivedKeysTransform
    derived_keys = {
        # State derivation
        "state.left_arm": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 0,
            "end": 3,
        },
        "state.left_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "observation.state",
            "start": 3,
            "end": 6,
            "from": "rpy",
            "to": "rotation_6d",
            "convention": "XYZ",
        },
        "state.left_gripper": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 6,
            "end": 7,
        },
        "state.right_arm": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 7,
            "end": 10,
        },
        "state.right_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "observation.state",
            "start": 10,
            "end": 13,
            "from": "rpy",
            "to": "rotation_6d",
            "convention": "XYZ",
        },
        "state.right_gripper": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 13,
            "end": 14,
        },
        # Action derivation
        "action.left_arm": {
            "type": "keep_from_origin",
            "source_key": "action",
            "start": 0,
            "end": 3,
        },
        "action.left_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "action",
            "start": 3,
            "end": 6,
            "from": "rpy",
            "to": "rotation_6d",
            "convention": "XYZ",
        },
        "action.left_gripper": {
            "type": "keep_from_origin",
            "source_key": "action",
            "start": 6,
            "end": 7,
        },
        "action.right_arm": {
            "type": "keep_from_origin",
            "source_key": "action",
            "start": 7,
            "end": 10,
        },
        "action.right_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "action",
            "start": 10,
            "end": 13,
            "from": "rpy",
            "to": "rotation_6d",
            "convention": "XYZ",
        },
        "action.right_gripper": {
            "type": "keep_from_origin",
            "source_key": "action",
            "start": 13,
            "end": 14,
        },
    }

    # Normalization modes (default, gripper can be overridden)
    default_normalization_modes = {
        "action.left_arm": "q99",
        "action.right_arm": "q99",
        "state.left_arm": "q99",
        "state.right_arm": "q99",
    }

    def _derived_keys_for(self, modality: str) -> dict[str, dict[str, Any]]:
        """Filter derived_keys for a specific modality."""
        prefix = f"{modality}."
        return {key: value for key, value in self.derived_keys.items() if key.startswith(prefix)}

    def modality_config(self, data_cfg=None):
        """Return modality configurations with dynamic action_chunk_size.

        Args:
            data_cfg: Optional config dict with:
                - action_chunk_size: int (default: 16)
        """
        cfg = data_cfg or {}

        # Get action chunk size
        action_chunk_size = int(cfg.get(
            "action_chunk_size",
            self.default_action_chunk_size
        ))

        # Generate action indices (2-step interval)
        action_indices = list(range(0, action_chunk_size * 2, 2))

        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.raw_state_keys,      # Raw: observation.state
                output_keys=self.state_keys,             # Derived: state.left_arm, etc.
                derived_keys=self._derived_keys_for("state"),
            ),
            "action": ModalityConfig(
                delta_indices=action_indices,               # Dynamic
                modality_keys=self.raw_action_keys,         # Raw: action
                output_keys=self.action_keys,               # Derived: action.left_arm, etc.
                derived_keys=self._derived_keys_for("action"),
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def video_transforms(self):
        """Video preprocessing transforms."""
        return [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
        ]

    def pose_transforms(self):
        """Pose derivation transforms (split vectors + RPY→rot6d)."""
        return [
            DerivedKeysTransform(
                apply_to=list(self.derived_keys.keys()),
                derived_keys=self.derived_keys,
                drop_source_keys=True,  # Remove raw keys after derivation
            )
        ]

    def transform(self, data_cfg=None):
        """Build complete transform pipeline.

        Args:
            data_cfg: Optional config dict with:
                - action_mode: str (default: "abs")
                - gripper_normalization: str or None (default: None)
                - gripper_binary_threshold: float (default: 0.5)
        """
        cfg = data_cfg or {}

        # Read configuration
        action_mode = str(cfg.get("action_mode", self.default_action_mode)).lower()
        if action_mode not in {"abs", "delta", "relative_pose"}:
            raise ValueError(f"action_mode must be 'abs'|'delta'|'relative_pose', got {action_mode!r}")

        gripper_norm = cfg.get("gripper_normalization", self.default_gripper_normalization)
        if gripper_norm is not None:
            gripper_norm = str(gripper_norm).lower()
            if gripper_norm not in {"binary", "min_max"}:
                raise ValueError(f"gripper_normalization must be None|'binary'|'min_max', got {gripper_norm!r}")

        gripper_binary_threshold = float(cfg.get("gripper_binary_threshold", self.default_gripper_binary_threshold))

        # Build normalization modes
        normalization_modes = self.default_normalization_modes.copy()

        # Add gripper normalization if specified
        if gripper_norm is not None:
            normalization_modes["action.left_gripper"] = gripper_norm
            normalization_modes["action.right_gripper"] = gripper_norm
            normalization_modes["state.left_gripper"] = gripper_norm
            normalization_modes["state.right_gripper"] = gripper_norm

        # Build transform pipeline
        state_action_keys = self.state_keys + self.action_keys

        transforms = self.video_transforms() + self.pose_transforms() + [
            StateActionToTensor(apply_to=state_action_keys),
        ]

        # Add action chunk transform (if not abs)
        if action_mode != "abs":
            transforms.append(
                ActionChunkTransform(
                    mode=action_mode,
                    apply_to=self.action_keys,
                    position_suffix="_arm",
                    rotation_suffix="_ori_6d",
                    gripper_suffix="_gripper",
                )
            )

        # Add normalization
        transforms.append(
            StateActionTransform(
                apply_to=state_action_keys,
                normalization_modes=normalization_modes,
                binary_threshold=gripper_binary_threshold,
            )
        )

        return ComposedModalityTransform(transforms=transforms)

    def transform_for_stats(self, data_cfg=None):
        """Build transform pipeline for stats computation (no normalization).

        Args:
            data_cfg: Optional config dict (same as transform())
        """
        cfg = data_cfg or {}

        action_mode = str(cfg.get("action_mode", self.default_action_mode)).lower()
        if action_mode not in {"abs", "delta", "relative_pose"}:
            raise ValueError(f"action_mode must be 'abs'|'delta'|'relative_pose', got {action_mode!r}")

        # Build transform pipeline (NO normalization)
        state_action_keys = self.state_keys + self.action_keys

        transforms = self.video_transforms() + self.pose_transforms() + [
            StateActionToTensor(apply_to=state_action_keys),
        ]

        # Add action chunk transform (if not abs)
        if action_mode != "abs":
            transforms.append(
                ActionChunkTransform(
                    mode=action_mode,
                    apply_to=self.action_keys,
                    position_suffix="_arm",
                    rotation_suffix="_ori_6d",
                    gripper_suffix="_gripper",
                )
            )

        # NO StateActionTransform here!

        return ComposedModalityTransform(transforms=transforms)

    def stats_path(self, data_cfg=None):
        """Get stats file path based on configuration.

        Args:
            data_cfg: Optional config dict with:
                - action_mode: str
                - action_chunk_size: int

        Returns:
            str: Relative path to stats file
        """
        cfg = data_cfg or {}

        action_mode = str(cfg.get("action_mode", self.default_action_mode)).lower()
        action_chunk_size = int(cfg.get("action_chunk_size", self.default_action_chunk_size))

        # Stats file naming convention
        return f"meta/stats_{action_mode}_chunk{action_chunk_size}.json"



# Export registry
ROBOT_TYPE_CONFIG_MAP = {
    "fastumi_dual_arm": FastUMIDualArmDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "fastumi_dual_arm": EmbodimentTag.NEW_EMBODIMENT,
}

# Dataset mixtures for training
DATASET_NAMED_MIXTURES = {
    # Single task for testing
    "single_task": [("Arrange_Toothbrush_and_Toothpaste", 1.0, "fastumi_dual_arm")],

    # All FastUMI tasks (commented out - enable when ready)
    # "all_fastumi": [(task, 1.0, "fastumi_dual_arm") for task in FASTUMI_DUAL_ARM_TASKS],
}
