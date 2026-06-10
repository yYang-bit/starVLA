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
    VideoResizeWithPad,
    VideoToNumpy,
    VideoToTensor,
)


class DerivedKeysTransform(ModalityTransform):
    """Create output keys from raw keys using keep_from_origin / transform_from_origin specs."""

    derived_keys: dict[str, dict[str, Any]] = Field(
        ...,
        description="Mapping from output key to source slicing or representation transform spec.",
    )
    drop_source_keys: bool = Field(default=True, description="Remove raw keys after all derived outputs are created.")

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "derived_keys", "drop_source_keys"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    @staticmethod
    def _as_tensor(value: Any) -> tuple[torch.Tensor, bool, torch.device | None, torch.dtype | None]:
        if isinstance(value, torch.Tensor):
            return value, True, value.device, value.dtype
        array = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(array), False, None, None

    @staticmethod
    def _restore_type(
        value: torch.Tensor,
        source_is_tensor: bool,
        source_device: torch.device | None,
        source_dtype: torch.dtype | None,
    ) -> Any:
        if source_is_tensor:
            assert source_device is not None and source_dtype is not None
            return value.to(device=source_device, dtype=source_dtype)
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _rotation_to_6d(value: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
        source_repr = str(spec.get("from", "")).lower()
        target_repr = str(spec.get("to", "")).lower()
        if target_repr != "rotation_6d":
            raise ValueError(f"Only to=rotation_6d is supported, got {target_repr}")

        if source_repr in {"rpy", "euler_rpy", "euler_angles_rpy"}:
            convention = str(spec.get("convention", "XYZ"))
            matrix = pt.euler_angles_to_matrix(value.to(torch.float32), convention=convention)
        elif source_repr in {"quat", "quaternion", "quaternion_xyzw", "quat_xyzw"}:
            quat_order = str(spec.get("quaternion_order", "xyzw")).lower()
            if quat_order == "xyzw":
                quat_wxyz = value[..., [3, 0, 1, 2]]
            elif quat_order == "wxyz":
                quat_wxyz = value
            else:
                raise ValueError(f"Unsupported quaternion_order={quat_order}")
            matrix = pt.quaternion_to_matrix(quat_wxyz.to(torch.float32))
        elif source_repr in {"rotation_matrix", "matrix"}:
            matrix = value.to(torch.float32).reshape(*value.shape[:-1], 3, 3)
        elif source_repr in {"rotation_6d", "rot6d"}:
            if value.shape[-1] != 6:
                raise ValueError(f"Expected rotation_6d source to have dim 6, got {tuple(value.shape)}")
            return value.to(torch.float32)
        else:
            raise ValueError(f"Unsupported rotation source representation: {source_repr}")

        return pt.matrix_to_rotation_6d(matrix)

    @staticmethod
    def _slice_source(source: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
        start = int(spec.get("start", 0))
        end = int(spec.get("end", source.shape[-1]))
        if start < 0 or end <= start or end > source.shape[-1]:
            raise ValueError(f"Invalid source slice [{start}:{end}] for shape {tuple(source.shape)}")
        return source[..., start:end]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        used_source_keys: set[str] = set()

        for output_key, spec in self.derived_keys.items():
            source_key = str(spec.get("source_key", ""))
            if source_key not in data:
                continue

            source, source_is_tensor, source_device, source_dtype = self._as_tensor(data[source_key])
            source = source.to(torch.float32)
            source_slice = self._slice_source(source, spec)
            spec_type = str(spec.get("type", "")).lower()

            if spec_type in {"keep_from_origin", "keep_from_orign"}:
                output = source_slice
            elif spec_type in {"transform_from_origin", "transform_from_orign"}:
                output = self._rotation_to_6d(source_slice, spec)
            else:
                raise ValueError(f"Unsupported derived key type for {output_key}: {spec.get('type')}")

            data[output_key] = self._restore_type(output, source_is_tensor, source_device, source_dtype)
            used_source_keys.add(source_key)

        if self.drop_source_keys:
            for source_key in used_source_keys:
                data.pop(source_key, None)

        return data


class MyDataConfig:
    video_keys = [
        "video.observation.images.camera_arm_left_upper_color",
        "video.observation.images.camera_arm_right_upper_color",
    ]
    raw_state_keys = [
        "observation.left_eef_pose",
        "observation.left_gripper_pos",
        "observation.right_eef_pose",
        "observation.right_gripper_pos",
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
    action_indices = list(range(0, 48, 3))
    state_indices = [0]

    derived_keys = {
        "state.left_arm": {"type": "keep_from_origin", "source_key": "observation.left_eef_pose", "start": 0, "end": 3},
        "state.left_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "observation.left_eef_pose",
            "start": 3,
            "end": 7,
            "from": "quaternion",
            "to": "rotation_6d",
            "quaternion_order": "xyzw",
        },
        "state.left_gripper": {
            "type": "keep_from_origin",
            "source_key": "observation.left_gripper_pos",
            "start": 0,
            "end": 1,
        },
        "state.right_arm": {"type": "keep_from_origin", "source_key": "observation.right_eef_pose", "start": 0, "end": 3},
        "state.right_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "observation.right_eef_pose",
            "start": 3,
            "end": 7,
            "from": "quaternion",
            "to": "rotation_6d",
            "quaternion_order": "xyzw",
        },
        "state.right_gripper": {
            "type": "keep_from_origin",
            "source_key": "observation.right_gripper_pos",
            "start": 0,
            "end": 1,
        },
        "action.left_arm": {"type": "keep_from_origin", "source_key": "action.left_eef_pose", "start": 0, "end": 3},
        "action.left_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "action.left_eef_pose",
            "start": 3,
            "end": 7,
            "from": "quaternion",
            "to": "rotation_6d",
            "quaternion_order": "xyzw",
        },
        "action.left_gripper": {
            "type": "keep_from_origin",
            "source_key": "action.left_gripper_pos",
            "start": 0,
            "end": 1,
        },
        "action.right_arm": {"type": "keep_from_origin", "source_key": "action.right_eef_pose", "start": 0, "end": 3},
        "action.right_ori_6d": {
            "type": "transform_from_origin",
            "source_key": "action.right_eef_pose",
            "start": 3,
            "end": 7,
            "from": "quaternion",
            "to": "rotation_6d",
            "quaternion_order": "xyzw",
        },
        "action.right_gripper": {
            "type": "keep_from_origin",
            "source_key": "action.right_gripper_pos",
            "start": 0,
            "end": 1,
        },
    }

    action_position_normalization_mode = "min_max"

    normalization_modes = {
        "action.left_arm": action_position_normalization_mode,
        "action.left_gripper": "min_max",
        "action.right_arm": action_position_normalization_mode,
        "action.right_gripper": "min_max",
    }

    def _derived_keys_for(self, modality: str) -> dict[str, dict[str, Any]]:
        prefix = f"{modality}."
        return {key: value for key, value in self.derived_keys.items() if key.startswith(prefix)}

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.raw_state_keys,
                output_keys=self.state_keys,
                derived_keys=self._derived_keys_for("state"),
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.raw_action_keys,
                output_keys=self.action_keys,
                derived_keys=self._derived_keys_for("action"),
            ),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def video_transforms(self):
        return [
            VideoToTensor(apply_to=self.video_keys),
            VideoResizeWithPad(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
        ]

    def pose_transforms(self):
        return [
            DerivedKeysTransform(
                apply_to=list(self.derived_keys),
                derived_keys=self.derived_keys,
                drop_source_keys=True,
            )
        ]

    def relative_pose_transform(self):
        return RelativePoseActionTransform(
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
        )

    def transform(self, data_cfg=None):
        action_repr = str((data_cfg or {}).get("action_chunk_representation", "")).lower()
        if action_repr == "relative_pose":
            return ComposedModalityTransform(
                transforms=self.video_transforms()
                + self.pose_transforms()
                + [
                    self.relative_pose_transform(),
                    StateActionToTensor(apply_to=self.action_keys),
                    StateActionTransform(
                        apply_to=self.action_keys,
                        normalization_modes=self.normalization_modes,
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
    "mymy_mix": [
        ("unknown", 1.0, "my_robot"),
    ],
}
