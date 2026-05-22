"""Galbot bimanual (relative-state) — data config, embodiment tags, and mixtures.

Dataset convention (assumed precomputed offline):
    observation.bimanual_relative : [T, 9]   rel_xyz(3) + rel_rot6d(6)
        Left arm pose expressed in right arm's base frame, no gripper.
    action                        : [T, 20]  per arm: xyz(3) + rot6d(6) + gripper(1), L||R.

Stats are loaded from <dataset>/meta/stats.json (precomputed offline).

Normalization policy:
    action.{left,right}_pos       : min_max
    action.{left,right}_ori_6d    : skip (rot6d is intrinsically in [-1, 1])
    action.{left,right}_gripper   : "binary" (default) or "min_max" via data_cfg.gripper_normalization
    state.bimanual_relative_pos   : min_max
    state.bimanual_relative_ori_6d: skip
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class GalbotBimanualRelativeDataConfig:
    video_keys = [
        "video.left_wrist",
        "video.right_wrist",
    ]

    state_keys = [
        "state.bimanual_relative_pos",
        "state.bimanual_relative_ori_6d",
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
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        gripper_norm = str((data_cfg or {}).get("gripper_normalization", "binary")).lower()
        if gripper_norm not in {"binary", "min_max"}:
            raise ValueError(
                f"gripper_normalization must be 'binary' or 'min_max', got {gripper_norm!r}"
            )

        action_norm_modes = {
            "action.left_pos": "min_max",
            "action.right_pos": "min_max",
            "action.left_gripper": gripper_norm,
            "action.right_gripper": gripper_norm,
        }

        state_norm_modes = {
            "state.bimanual_relative_pos": "min_max",
        }

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=action_norm_modes,
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=state_norm_modes,
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "galbot_bimanual_relative": GalbotBimanualRelativeDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_bimanual_relative": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_mix": [
        ("<your_dataset_dir_name>", 1.0, "galbot_bimanual_relative"),
    ],
}
