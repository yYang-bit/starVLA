"""Galbot bimanual — data config, embodiment tags, and mixtures.

Dataset convention (abs-only in parquet, all transforms done at training time):
    action                         : [T, 20]  next-step abs eef pose + gripper
    observation.state              : [T, 20]  current-step abs eef pose + gripper
    observation.dual_relative_pose : [T, 9]   left arm in right arm's base frame (rel_xyz + rel_rot6d)

Action transform (controlled by data_cfg.action_mode):
    "abs"             : action[t] unchanged (world frame)
    "delta"           : Frame-to-frame delta in local end-effector frame
                        pos: delta_xyz[t] = R[t-1].T @ (p[t] - p[t-1]), delta_xyz[0] = 0
                        rot: delta_R[t] = R[t-1].T @ R[t], delta_R[0] = I
    "relative_pose"   : Relative to base (action[0]) in local end-effector frame
                        pos: rel_xyz[t] = R[0].T @ (p[t] - p[0])
                        rot: rel_R[t] = R[0].T @ R[t]

Local frame representation supports cross-embodiment learning by encoding
task-level motion patterns rather than absolute positions.

Configuration (all optional, defaults shown):
    action_mode: "abs"                     # Transform mode
    action_chunk_size: 30                  # Action chunk horizon
    gripper_normalization: None            # None (use raw values) | "binary" | "min_max"
    gripper_binary_threshold: 100.0        # Threshold for binary mode (mm)

Stats file naming convention:
    abs mode:           meta/stats_abs_chunk{size}.json
    delta mode:         meta/stats_delta_chunk{size}.json
    relative_pose mode: meta/stats_relative_chunk{size}.json

Normalization policy:
    action.{left,right}_pos                  : q99 → [-1, 1]
    action.{left,right}_ori_6d               : skip (rot6d is intrinsically in [-1, 1])
    action.{left,right}_gripper              : optional (binary/min_max or raw)
    state.dual_relative_pose_pos             : q99 → [-1, 1]
    state.dual_relative_pose_ori_6d          : skip
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.action_chunk_mode import (
    ActionChunkTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class GalbotBimanualSelfDataConfig:
    """Galbot bimanual data config with unified action chunk transform.

    Flow: StateActionToTensor → ActionChunkTransform → StateActionTransform (normalize)
    """

    # Default configuration (can be overridden by data_cfg)
    default_action_mode = "abs"
    default_action_chunk_size = 30
    default_gripper_normalization = None  # None = use raw values
    default_gripper_binary_threshold = 100.0

    # Fixed dataset properties
    video_keys = [
        "video.left_wrist",
        "video.right_wrist",
    ]

    state_keys = [
        "state.dual_relative_pose_pos",
        "state.dual_relative_pose_ori_6d",
    ]

    action_keys = [
        "action.left_pos",
        "action.left_ori_6d",
        "action.left_gripper",
        "action.right_pos",
        "action.right_ori_6d",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    # Observation/state indices (fixed)
    observation_indices = [0]
    state_indices = [0]
    # action_indices is dynamic (generated in modality_config)

    def modality_config(self, data_cfg=None):
        """Build modality config with dynamic action_chunk_size.

        Args:
            data_cfg: Optional config dict with:
                - action_chunk_size: int (default: 30)
        """
        cfg = data_cfg or {}

        # Get action chunk size from config
        action_chunk_size = int(cfg.get(
            "action_chunk_size",
            self.default_action_chunk_size
        ))

        # Generate action indices dynamically
        action_indices = list(range(action_chunk_size))

        return {
            "video":    ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state":    ModalityConfig(delta_indices=self.state_indices,      modality_keys=self.state_keys),
            "action":   ModalityConfig(delta_indices=action_indices,          modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        """Build complete transform pipeline.

        Args:
            data_cfg: Optional config dict with:
                - action_mode: str (default: "abs")
                - gripper_normalization: str or None (default: None)
                - gripper_binary_threshold: float (default: 100.0)
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
        action_norm_modes = {
            "action.left_pos":      "q99",
            "action.right_pos":     "q99",
        }

        # Add gripper normalization if specified
        if gripper_norm is not None:
            action_norm_modes["action.left_gripper"] = gripper_norm
            action_norm_modes["action.right_gripper"] = gripper_norm

        state_norm_modes = {
            "state.dual_relative_pose_pos": "q99",
        }

        # Build transform pipeline
        transforms = [
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
        ]

        # Add action chunk transform (if not abs)
        if action_mode != "abs":
            transforms.append(
                ActionChunkTransform(
                    mode=action_mode,
                    action_keys=self.action_keys,
                    position_suffix="_pos",
                    rotation_suffix="_ori_6d",
                    gripper_suffix="_gripper",
                )
            )

        # Add normalization
        transforms.extend([
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=action_norm_modes,
                binary_threshold=gripper_binary_threshold,
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=state_norm_modes,
            ),
        ])

        return ComposedModalityTransform(transforms=transforms)

    def transform_for_stats(self, data_cfg=None):
        """Build transform pipeline for stats computation (no normalization).

        IMPORTANT: Stats are computed on transformed data (after action chunk transform).
        This ensures stats match the actual data distribution used in training.

        Args:
            data_cfg: Optional config dict (same as transform())
        """
        cfg = data_cfg or {}

        action_mode = str(cfg.get("action_mode", self.default_action_mode)).lower()
        if action_mode not in {"abs", "delta", "relative_pose"}:
            raise ValueError(f"action_mode must be 'abs'|'delta'|'relative_pose', got {action_mode!r}")

        # Build transform pipeline (NO normalization)
        transforms = [
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
        ]

        # Add action chunk transform (if not abs)
        if action_mode != "abs":
            transforms.append(
                ActionChunkTransform(
                    mode=action_mode,
                    action_keys=self.action_keys,
                    position_suffix="_pos",
                    rotation_suffix="_ori_6d",
                    gripper_suffix="_gripper",
                )
            )

        # NO StateActionTransform here!
        # Stats are computed on transformed raw data

        return ComposedModalityTransform(transforms=transforms)

    def stats_path(self, data_cfg=None):
        """Get stats file path based on configuration.

        Args:
            data_cfg: Optional config dict with:
                - action_mode: str
                - action_chunk_size: int

        Returns:
            str: Relative path to stats file (e.g., "meta/stats_delta_chunk30.json")
        """
        cfg = data_cfg or {}

        action_mode = str(cfg.get("action_mode", self.default_action_mode)).lower()
        action_chunk_size = int(cfg.get("action_chunk_size", self.default_action_chunk_size))

        # Stats file naming convention
        return f"meta/stats_{action_mode}_chunk{action_chunk_size}.json"


# ─── Registry ───────────────────────────────────────────────────────────────

ROBOT_TYPE_CONFIG_MAP = {
    "galbot_bimanual_self": GalbotBimanualSelfDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_bimanual_self": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_self_mix": [
        # data_name is overridden at runtime via --datasets.vla_data.data_name in .sh
        ("galbot_lerobot_dual_cup", 1.0, "galbot_bimanual_self"),
    ],
}
