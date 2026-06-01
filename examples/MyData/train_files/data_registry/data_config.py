from typing import Any

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    RelativePoseActionTransform,
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


class PoseQuatToRotation6DTransform(ModalityTransform):
    """Derive position and 6D rotation keys from [x, y, z, qx, qy, qz, qw] pose keys."""

    output_map: dict[str, tuple[str, str]] = Field(
        ...,
        description="Mapping from source pose key to (position_output_key, rotation_6d_output_key).",
    )
    drop_source_keys: bool = Field(default=False, description="Remove source pose keys after creating output keys.")

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "output_map", "drop_source_keys"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    @staticmethod
    def _as_tensor(value: Any) -> tuple[torch.Tensor, bool, torch.device | None, torch.dtype | None]:
        if isinstance(value, torch.Tensor):
            return value, True, value.device, value.dtype
        array = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(array), False, None, None

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for source_key in self.apply_to:
            if source_key not in data:
                continue
            if source_key not in self.output_map:
                raise KeyError(f"Missing output_map entry for {source_key}")

            position_key, rotation_key = self.output_map[source_key]
            pose, source_is_tensor, source_device, source_dtype = self._as_tensor(data[source_key])
            pose = pose.to(torch.float32)
            if pose.shape[-1] != 7:
                raise ValueError(f"Expected pose key {source_key} to have shape (..., 7), got {tuple(pose.shape)}")

            position = pose[..., :3]
            quat_xyzw = pose[..., 3:7]
            quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
            rotation_matrix = pt.quaternion_to_matrix(quat_wxyz)
            rotation_6d = rotation_matrix[..., :2, :].reshape(*rotation_matrix.shape[:-2], 6)

            if source_is_tensor:
                assert source_device is not None and source_dtype is not None
                data[position_key] = position.to(device=source_device, dtype=source_dtype)
                data[rotation_key] = rotation_6d.to(device=source_device, dtype=source_dtype)
            else:
                data[position_key] = position.detach().cpu().numpy().astype(np.float32, copy=False)
                data[rotation_key] = rotation_6d.detach().cpu().numpy().astype(np.float32, copy=False)

            if self.drop_source_keys:
                data.pop(source_key, None)

        return data


class CopyKeysTransform(ModalityTransform):
    """Copy raw keys to their post-transform names."""

    output_map: dict[str, str] = Field(..., description="Mapping from source key to copied output key.")
    drop_source_keys: bool = Field(default=False, description="Remove source keys after copying.")

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "output_map", "drop_source_keys"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for source_key in self.apply_to:
            if source_key not in data:
                continue
            if source_key not in self.output_map:
                raise KeyError(f"Missing output_map entry for {source_key}")
            data[self.output_map[source_key]] = data[source_key]
            if self.drop_source_keys:
                data.pop(source_key, None)
        return data


class MyDataConfig:
    video_keys = [
        "video.observation.images.camera_arm_left_upper_color",
        "video.observation.images.camera_arm_right_upper_color",
    ]
    raw_state_keys = [
        "state.left_eef_pose",
        "state.left_gripper_pos",
        "state.right_eef_pose",
        "state.right_gripper_pos",
    ]
    raw_action_keys = [
        "action.left_eef_pose",
        "action.left_gripper_pos",
        "action.right_eef_pose",
        "action.right_gripper_pos",
    ]
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

    observation_indices = [0]
    action_indices = list(range(0, 16))
    state_indices = [0]

    pose_output_map = {
        "state.left_eef_pose": ("state.left_arm", "state.left_ori_6d"),
        "state.right_eef_pose": ("state.right_arm", "state.right_ori_6d"),
        "action.left_eef_pose": ("action.left_arm", "action.left_ori_6d"),
        "action.right_eef_pose": ("action.right_arm", "action.right_ori_6d"),
    }

    copy_output_map = {
        "state.left_gripper_pos": "state.left_gripper",
        "state.right_gripper_pos": "state.right_gripper",
        "action.left_gripper_pos": "action.left_gripper",
        "action.right_gripper_pos": "action.right_gripper",
    }

    derived_keys = {
        "state.left_arm": {"type": "pose_quat_position", "source_key": "state.left_eef_pose"},
        "state.left_ori_6d": {"type": "pose_quat_rotation_6d", "source_key": "state.left_eef_pose"},
        "state.left_gripper": {"type": "copy", "source_key": "state.left_gripper_pos"},
        "state.right_arm": {"type": "pose_quat_position", "source_key": "state.right_eef_pose"},
        "state.right_ori_6d": {"type": "pose_quat_rotation_6d", "source_key": "state.right_eef_pose"},
        "state.right_gripper": {"type": "copy", "source_key": "state.right_gripper_pos"},
        "action.left_arm": {"type": "pose_quat_position", "source_key": "action.left_eef_pose"},
        "action.left_ori_6d": {"type": "pose_quat_rotation_6d", "source_key": "action.left_eef_pose"},
        "action.left_gripper": {"type": "copy", "source_key": "action.left_gripper_pos"},
        "action.right_arm": {"type": "pose_quat_position", "source_key": "action.right_eef_pose"},
        "action.right_ori_6d": {"type": "pose_quat_rotation_6d", "source_key": "action.right_eef_pose"},
        "action.right_gripper": {"type": "copy", "source_key": "action.right_gripper_pos"},
    }

    normalization_modes = {
        "action.left_arm": "min_max",
        "action.left_gripper": "min_max",
        "action.right_arm": "min_max",
        "action.right_gripper": "min_max",
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.raw_state_keys,
                output_keys=self.state_keys,
                derived_keys={
                    key: value for key, value in self.derived_keys.items() if key.startswith("state.")
                },
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.raw_action_keys,
                output_keys=self.action_keys,
                derived_keys={
                    key: value for key, value in self.derived_keys.items() if key.startswith("action.")
                },
            ),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }
    def video_transforms(self):
        return [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
        ]

#pose_output_map
    def pose_transforms(self):
        return [
            PoseQuatToRotation6DTransform(
                apply_to=list(self.pose_output_map.keys()),
                output_map=self.pose_output_map,
                drop_source_keys=True,
            ),
            CopyKeysTransform(
                apply_to=list(self.copy_output_map.keys()),
                output_map=self.copy_output_map,
                drop_source_keys=True,
            ),
        ]

    def transform(self, data_cfg=None):
        action_repr = str((data_cfg or {}).get("action_chunk_representation", "")).lower()
        if action_repr == "relative_pose":
            return ComposedModalityTransform(
                transforms=self.video_transforms()
                + self.pose_transforms()
                + [
                    RelativePoseActionTransform(
                        apply_to=self.state_keys + self.action_keys,
                        state_keys=self.state_keys,
                        action_keys=self.action_keys,
                        arm_prefixes=["left", "right"],
                        state_position_suffix="_arm",
                        state_rotation_suffix="_ori_6d",
                        state_gripper_suffix="_gripper",
                        action_position_suffix="_arm",
                        action_rotation_suffix="_ori_6d",
                        action_gripper_suffix="_gripper",
                    ),
                    StateActionToTensor(apply_to=self.action_keys),
                    StateActionTransform(
                        apply_to=self.action_keys,
                        normalization_modes={
                            key: value
                            for key, value in self.normalization_modes.items()
                            if key.startswith("action.")
                        },
                    ),
                ]
            )

        state_action_keys = self.state_keys + self.action_keys
        return ComposedModalityTransform(
            transforms=self.video_transforms()
            + self.pose_transforms()
            + [
                StateActionToTensor(apply_to=state_action_keys),
                StateActionTransform(
                    apply_to=state_action_keys,
                    normalization_modes=self.normalization_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "my_robot": MyDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "my_robot": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "my_mix": [
        ("lerobot", 1.0, "my_robot"),
    ],
}
