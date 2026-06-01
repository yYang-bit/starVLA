"""FastUMI-100K data config, embodiment tags, and example mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    RpyToRotation6DTransform,
    StateActionToTensor,
    StateActionTransform,
)


def _rpy_group(modality: str, prefix: str = "") -> list[str]:
    stem = f"{prefix}_" if prefix else ""
    return [
        f"{modality}.{stem}roll",
        f"{modality}.{stem}pitch",
        f"{modality}.{stem}yaw",
    ]


class FastUMISingleArmDataConfig:
    video_keys = ["video.primary_image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    state_output_keys = ["state.x", "state.y", "state.z", "state.rotation_6d", "state.gripper"]
    action_output_keys = ["action.x", "action.y", "action.z", "action.rotation_6d", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
                output_keys=self.state_output_keys,
                derived_keys={
                    "state.rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("state"),
                    }
                },
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
                output_keys=self.action_output_keys,
                derived_keys={
                    "action.rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("action"),
                    }
                },
            ),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        output_keys = self.state_output_keys + self.action_output_keys
        return ComposedModalityTransform(
            transforms=[
                RpyToRotation6DTransform(
                    apply_to=self.state_keys + self.action_keys,
                    groups={
                        "state.rotation_6d": _rpy_group("state"),
                        "action.rotation_6d": _rpy_group("action"),
                    },
                ),
                StateActionToTensor(apply_to=output_keys),
                StateActionTransform(
                    apply_to=output_keys,
                    normalization_modes={key: "min_max" for key in output_keys},
                ),
            ]
        )


class FastUMIDualArmDataConfig:
    video_keys = ["video.primary_image"]
    state_keys = [
        "state.left_x",
        "state.left_y",
        "state.left_z",
        "state.left_roll",
        "state.left_pitch",
        "state.left_yaw",
        "state.left_gripper",
        "state.right_x",
        "state.right_y",
        "state.right_z",
        "state.right_roll",
        "state.right_pitch",
        "state.right_yaw",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_x",
        "action.left_y",
        "action.left_z",
        "action.left_roll",
        "action.left_pitch",
        "action.left_yaw",
        "action.left_gripper",
        "action.right_x",
        "action.right_y",
        "action.right_z",
        "action.right_roll",
        "action.right_pitch",
        "action.right_yaw",
        "action.right_gripper",
    ]
    state_output_keys = [
        "state.left_x",
        "state.left_y",
        "state.left_z",
        "state.left_rotation_6d",
        "state.left_gripper",
        "state.right_x",
        "state.right_y",
        "state.right_z",
        "state.right_rotation_6d",
        "state.right_gripper",
    ]
    action_output_keys = [
        "action.left_x",
        "action.left_y",
        "action.left_z",
        "action.left_rotation_6d",
        "action.left_gripper",
        "action.right_x",
        "action.right_y",
        "action.right_z",
        "action.right_rotation_6d",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
                output_keys=self.state_output_keys,
                derived_keys={
                    "state.left_rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("state", "left"),
                    },
                    "state.right_rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("state", "right"),
                    },
                },
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
                output_keys=self.action_output_keys,
                derived_keys={
                    "action.left_rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("action", "left"),
                    },
                    "action.right_rotation_6d": {
                        "type": "rpy_to_rotation_6d",
                        "source_keys": _rpy_group("action", "right"),
                    },
                },
            ),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        output_keys = self.state_output_keys + self.action_output_keys
        return ComposedModalityTransform(
            transforms=[
                RpyToRotation6DTransform(
                    apply_to=self.state_keys + self.action_keys,
                    groups={
                        "state.left_rotation_6d": _rpy_group("state", "left"),
                        "state.right_rotation_6d": _rpy_group("state", "right"),
                        "action.left_rotation_6d": _rpy_group("action", "left"),
                        "action.right_rotation_6d": _rpy_group("action", "right"),
                    },
                ),
                StateActionToTensor(apply_to=output_keys),
                StateActionTransform(
                    apply_to=output_keys,
                    normalization_modes={key: "min_max" for key in output_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "fastumi_single_arm": FastUMISingleArmDataConfig(),
    "fastumi_dual_arm": FastUMIDualArmDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "fastumi_single_arm": EmbodimentTag.NEW_EMBODIMENT,
    "fastumi_dual_arm": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "fastumi_dual_arm": [("dual_arm", 1.0, "fastumi_dual_arm")],
}
