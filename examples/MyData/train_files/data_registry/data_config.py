from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    RelativePoseActionTransform,
    StateActionToTensor,
    StateActionTransform,
)


class MyDataConfig:
    video_keys = [
        "video.cam_head_l_img",
        "video.left_arm_camera",
        "video.right_arm_camera",
    ]
    state_keys = [
        "state.left_abs_pos",
        "state.left_abs_ori_6d",
        "state.left_gripper",
        "state.right_abs_pos",
        "state.right_abs_ori_6d",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_abs_pos",
        "action.left_abs_ori_6d",
        "action.left_gripper",
        "action.right_abs_pos",
        "action.right_abs_ori_6d",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))
    state_indices = [-1]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        action_repr = str((data_cfg or {}).get("action_chunk_representation", "")).lower()
        if action_repr == "relative_pose":
            return ComposedModalityTransform(transforms=[
                RelativePoseActionTransform(
                    apply_to=self.state_keys + self.action_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    arm_prefixes=["left", "right"],
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.left_abs_pos": "q99",
                        "action.left_gripper": "min_max",
                        "action.right_abs_pos": "q99",
                        "action.right_gripper": "min_max",
                    },
                ),
            ])

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_abs_pos": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_abs_pos": "min_max",
                    "action.right_gripper": "binary",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "my_robot": MyDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "my_robot": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "my_mix": [
        ("20260412", 1.0, "my_robot"),
    ],
}
