"""Galbot bimanual (self-mode) — data config, embodiment tags, and mixtures.

Dataset convention (abs-only in parquet, all transforms done at training time):
    action                         : [T, 20]  next-step abs eef pose + gripper
    observation.state              : [T, 20]  current-step abs eef pose + gripper
    observation.dual_relative_pose : [T, 9]   left arm in right arm's base frame (rel_xyz + rel_rot6d)

Action transform (controlled by data_cfg.self_mode):
    "abs"             : action[t] unchanged
    "delta"           : action[t] -= action[t-1], action[0] = 0
    "chunk_relative"  : action[t] -= action[0]

Stats are loaded from <dataset>/meta/stats.json (precomputed offline by compute_galbot_stats_self_mode.py).

Normalization policy:
    action.{left,right}_pos                  : q99 → [-1, 1]
    action.{left,right}_ori_6d               : skip (rot6d is intrinsically in [-1, 1])
    action.{left,right}_gripper              : binary (threshold=100mm) → {0, 1}
    state.dual_relative_pose_pos             : q99 → [-1, 1]
    state.dual_relative_pose_ori_6d          : skip

Gripper binary threshold (100mm) controlled by data_cfg.gripper_binary_threshold.
Inference post-processing: model output 0 → 46mm, 1 → 124mm (handled outside training).
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


class GalbotBimanualSelfDataConfig:
    """Self-mode data config — action transform and stats both computed offline.

    Flow: StateActionToTensor → SelfModeActionTransform → StateActionTransform (normalize)
    """

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

    observation_indices = [0]
    state_indices = [0]
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
        state_norm_modes = {
            "state.dual_relative_pose_pos": "q99",
        }

        # Gripper should NOT be differenced — binary threshold applies to abs values
        action_keys_no_gripper = [
            "action.left_pos",
            "action.left_ori_6d",
            "action.right_pos",
            "action.right_ori_6d",
        ]

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
            # --- self-mode action transform (before normalization) ---
            # IMPORTANT: exclude gripper from delta — binary threshold needs abs values
            SelfModeActionTransform(
                apply_to=action_keys_no_gripper,
                self_mode=self_mode,
            ),
            # --- normalization ---
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


# ─── Registry ───────────────────────────────────────────────────────────────

ROBOT_TYPE_CONFIG_MAP = {
    "galbot_bimanual_self": GalbotBimanualSelfDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_bimanual_self": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_self_mix": [
        ("galbot_lerobot_dual_cup_0526", 1.0, "galbot_bimanual_self"),
    ],
}
