"""Galbot bimanual (self-mode, NO STATE) — data config for contrast experiment.

This is a variant of the original config with state input REMOVED.
Only vision (dual wrist cameras) is used as input.

Dataset convention (same as original):
    action                         : [T, 20]  next-step abs eef pose + gripper
    observation.state              : [T, 20]  NOT USED (not loaded)
    observation.dual_relative_pose : [T, 9]   NOT USED (not loaded)

Action transform (controlled by data_cfg.self_mode):
    "abs"             : action[t] unchanged
    "delta"           : pos: action[t] -= action[t-1], action[0] = 0
                        rotation_6d: SO(3) relative rotation R[t] @ R[t-1]^T, R[0] = I
    "chunk_relative"  : pos: action[t] -= action[0]
                        rotation_6d: SO(3) relative rotation R[t] @ R[0]^T

Stats are loaded from <dataset>/meta/stats.json (same as original).

Normalization policy (same as original):
    action.{left,right}_pos                  : q99 → [-1, 1]
    action.{left,right}_ori_6d               : skip (rot6d is intrinsically in [-1, 1])
    action.{left,right}_gripper              : binary (threshold=100mm) → {0, 1}
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.self_mode_action import (
    SelfModeActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class GalbotBimanualSelfNoStateDataConfig:
    """Self-mode data config WITHOUT state input — vision-only baseline.

    Flow: StateActionToTensor → SelfModeActionTransform → StateActionTransform (normalize)
    """

    video_keys = [
        "video.left_wrist",
        "video.right_wrist",
    ]

    state_keys = []  # EMPTY - no state input for this contrast experiment

    action_keys = [
        "action.left_pos",
        "action.left_ori_6d",
        "action.left_gripper",
        "action.right_pos",
        "action.right_ori_6d",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]  # Not used since state_keys is empty
    action_indices = list(range(30))

    def modality_config(self):
        return {
            "video":    ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state":    ModalityConfig(delta_indices=self.state_indices,      modality_keys=self.state_keys),
            "action":   ModalityConfig(delta_indices=self.action_indices,     modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        cfg = data_cfg or {}

        self_mode = str(cfg.get("self_mode", "abs")).lower()
        if self_mode not in {"abs", "delta", "chunk_relative"}:
            raise ValueError(f"self_mode must be 'abs' / 'delta' / 'chunk_relative', got {self_mode!r}")

        gripper_norm = str(cfg.get("gripper_normalization", "binary")).lower()
        if gripper_norm not in {"binary", "min_max"}:
            raise ValueError(f"gripper_normalization must be 'binary' or 'min_max', got {gripper_norm!r}")

        gripper_binary_threshold = float(cfg.get("gripper_binary_threshold", 100.0))

        action_norm_modes = {
            "action.left_pos":      "q99",
            "action.right_pos":     "q99",
            "action.left_gripper":  gripper_norm,
            "action.right_gripper": gripper_norm,
        }
        # No state normalization needed (state_keys is empty)

        # Gripper should NOT be differenced — binary threshold applies to abs values
        action_keys_no_gripper = [
            "action.left_pos",
            "action.left_ori_6d",
            "action.right_pos",
            "action.right_ori_6d",
        ]

        # Rotation keys require SO(3) operations (not arithmetic difference)
        rotation_keys = [
            "action.left_ori_6d",
            "action.right_ori_6d",
        ]

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),  # No state_keys
            # --- self-mode action transform (before normalization) ---
            # IMPORTANT: exclude gripper from delta — binary threshold needs abs values
            # rotation_keys use SO(3) relative rotation, pos keys use arithmetic difference
            SelfModeActionTransform(
                apply_to=action_keys_no_gripper,
                self_mode=self_mode,
                rotation_keys=rotation_keys,
            ),
            # --- normalization ---
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=action_norm_modes,
                binary_threshold=gripper_binary_threshold,
            ),
            # No state normalization transform needed
        ])


# ─── Registry ───────────────────────────────────────────────────────────────

ROBOT_TYPE_CONFIG_MAP = {
    "galbot_bimanual_self_no_state": GalbotBimanualSelfNoStateDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_bimanual_self_no_state": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_self_no_state_mix": [
        # data_name is overridden at runtime via --datasets.vla_data.data_name in .sh
        ("galbot_lerobot_dual_cup", 1.0, "galbot_bimanual_self_no_state"),
    ],
}
