"""
StarVLA Policy Server Dual-Arm Inference Example — Real Code + Pseudocode

This script shows how to connect to a StarVLA policy server over WebSocket,
receive model-predicted dual-arm actions, unnormalize them, and execute them on a dual-arm robot.

⚠️ Note: the current implementation is based on a dual-arm Franka setup with a 14D action space:
    [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l,
     x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]
    where action[0:7] corresponds to the left arm and action[7:14] to the right arm.

For other dual-arm robots, the action dimensionality, arm ordering, and gripper indices may differ.
Please adjust `action_dim`, gripper indices, and unnormalization logic to match your robot.

For single-arm robots (7D), see `inference_single_example.py`.

Real code sections (directly reusable):
    - WebSocket client connection and communication
    - request construction and response parsing
    - action unnormalization (`unnormalize_actions`), supporting 7D single-arm and 14D dual-arm actions

Pseudocode sections (replace for your robot):
    - camera image acquisition
    - dual-arm robot environment creation, `reset`, and `step`
"""

import json
from typing import Dict, List

import numpy as np
import pytorch3d.transforms as pt
import torch
from deployment.model_server.tools.show_video import VideoStreamer
from deployment.model_server.tools.websocket_server import SimpleWebsocketServer
from starVLA.model.framework.base_framework import baseframework


# ============================================================
# ✅ Real code: action unnormalization (supports 7D single-arm and 14D dual-arm)
# ============================================================
def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalized model outputs in [-1, 1] back to the real action space.
    Supports both single-arm (7D) and dual-arm (14D) action spaces.

    Args:
        normalized_actions: normalized actions, shape [T, action_dim], range [-1, 1]
            - single-arm: action_dim=7, gripper at index=6
            - dual-arm: action_dim=14, grippers at index=6 (left arm) and index=13 (right arm)
        action_norm_stats: normalization statistics containing:
            - "q01": np.ndarray, per-dimension 1st percentile values
            - "q99": np.ndarray, per-dimension 99th percentile values
            - "mask": np.ndarray (bool), which dimensions are normalized

    Returns:
        actions: unnormalized actions, shape [T, action_dim]

    Formula:
        action = 0.5 * (normalized + 1) * (q99 - q01) + q01
        where the gripper dimensions are first binarized: < 0.5 → -1, >= 0.5 → 1
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    normalized_actions = np.clip(normalized_actions, -1, 1)

    # Linear unnormalization, only for dimensions with mask=True.
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


# ============================================================
# ✅ Real code: request construction and response parsing
# ============================================================
def build_request(images: List[np.ndarray], task_instruction: str) -> dict:
    """
    Construct a request for the StarVLA policy server.

    Args:
        images: multi-view camera images, each with shape (H, W, 3), dtype uint8
        task_instruction: natural-language task instruction

    Returns:
        request_data: dict matching the server API
    """
    request_data = {
        "examples": [
            {
                "image": images,  # List[np.ndarray], converted to PIL Images on the server side
                "lang": task_instruction,
            }
        ]
    }
    return request_data


def parse_response(result: dict) -> np.ndarray:
    """
    Parse the policy server response and extract the action chunk.

    Args:
        result: dict returned by the server, formatted as:
            {"data": {"normalized_actions": np.ndarray}, "status": "ok"}

    Returns:
        action_chunk: shape [T, action_dim], where T is the predicted horizon
    """
    data = result.get("data", result)

    for key in ["normalized_actions", "actions", "action"]:
        if key in data:
            actions = data[key]
            if isinstance(actions, list):
                actions = np.array(actions)
            # Normalize to [T, action_dim].
            if len(actions.shape) == 3:
                actions = actions[0]  # [B, T, D] -> [T, D]
            elif len(actions.shape) == 1:
                actions = actions.reshape(1, -1)  # [D] -> [1, D]
            return actions

    raise KeyError(f"Could not extract actions from response. Available keys: {list(data.keys())}")


# ============================================================
# ✅ Real code: load action normalization statistics
# ============================================================
def load_action_norm_stats(json_path: str, embodiment_key: str = "new_embodiment") -> Dict[str, np.ndarray]:
    """
    Load normalization statistics from `dataset_statistics.json`.

    Example JSON format for a dual-arm 14D setup:
    {
        "new_embodiment": {
            "action": {
                "min": [14 minimum values],
                "max": [14 maximum values],
                "mask": [true, true, ..., true]  // 14 entries
            }
        }
    }
    """
    with open(json_path, "r") as f:
        stats_data = json.load(f)

    if "action_stats" in stats_data:
        action_stats = stats_data["action_stats"]
        low = np.zeros(20, dtype=np.float32)
        high = np.zeros(20, dtype=np.float32)
        mask = np.zeros(20, dtype=bool)

        def fill_min_max(start: int, end: int, stats_key: str):
            stats = action_stats[stats_key]
            stat_min = np.asarray(stats["min"], dtype=np.float32).reshape(-1)
            stat_max = np.asarray(stats["max"], dtype=np.float32).reshape(-1)
            width = end - start
            if stat_min.size == 1 and width > 1:
                stat_min = np.repeat(stat_min, width)
            if stat_max.size == 1 and width > 1:
                stat_max = np.repeat(stat_max, width)
            if stat_min.size != width or stat_max.size != width:
                raise ValueError(
                    f"Relative action stats `{stats_key}` shape mismatch: expected {width}, "
                    f"got min={stat_min.shape}, max={stat_max.shape}"
                )
            low[start:end] = stat_min
            high[start:end] = stat_max
            mask[start:end] = True

        fill_min_max(0, 3, "left_arm")
        fill_min_max(9, 10, "left_gripper")
        fill_min_max(10, 13, "right_arm")
        fill_min_max(19, 20, "right_gripper")
        return {"q01": low, "q99": high, "mask": mask}

    if embodiment_key in stats_data:
        stats_data = stats_data[embodiment_key]
    if "action" in stats_data:
        stats_data = stats_data["action"]

    norm_stats = {
        "q01": np.array(stats_data.get("q01", stats_data.get("low", []))),
        "q99": np.array(stats_data.get("q99", stats_data.get("high", []))),
    }
    if "mask" in stats_data:
        norm_stats["mask"] = np.array(stats_data["mask"], dtype=bool)

    return norm_stats


def build_local_policy(ckpt_path: str, use_bf16: bool = False):
    policy = baseframework.from_pretrained(ckpt_path)
    if use_bf16:
        policy = policy.to(torch.bfloat16)
    return policy.to("cuda").eval()


def inverse_action(action_chunk: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    """
    Convert dual-arm relative-pose actions back to absolute actions using current proprio.

    Expected layout for both `action_chunk` and `proprio`:
        left : [x, y, z, rot6d(6), gripper]  -> 10 dims
        right: [x, y, z, rot6d(6), gripper]  -> 10 dims

    Args:
        action_chunk: [T, 20] relative action chunk
        proprio: [20] or [1, 20] current absolute proprio

    Returns:
        absolute_action_chunk: [T, 20]
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    proprio = np.asarray(proprio, dtype=np.float32)

    if action_chunk.ndim != 2 or action_chunk.shape[-1] != 20:
        raise ValueError(f"Expected action_chunk shape [T, 20], got {action_chunk.shape}")
    if proprio.ndim == 2:
        if proprio.shape[0] != 1:
            raise ValueError(f"Expected proprio shape [20] or [1, 20], got {proprio.shape}")
        proprio = proprio[0]
    if proprio.ndim != 1 or proprio.shape[0] != 20:
        raise ValueError(f"Expected proprio shape [20] or [1, 20], got {proprio.shape}")

    def _arm_slices(offset: int):
        pos_slice = slice(offset, offset + 3)
        rot_slice = slice(offset + 3, offset + 9)
        grip_slice = slice(offset + 9, offset + 10)
        return pos_slice, rot_slice, grip_slice

    abs_components = []
    for arm_offset in (0, 10):
        pos_slice, rot_slice, grip_slice = _arm_slices(arm_offset)

        init_pos = proprio[pos_slice]
        init_rot_6d = proprio[rot_slice]
        init_rot = (
            pt.rotation_6d_to_matrix(torch.from_numpy(init_rot_6d[None, :]))
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        rel_pos = action_chunk[:, pos_slice]
        rel_rot_6d = action_chunk[:, rot_slice]
        rel_rot = (
            pt.rotation_6d_to_matrix(torch.from_numpy(rel_rot_6d))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        abs_pos = (init_rot @ rel_pos.T).T + init_pos[None, :]
        abs_rot = np.einsum("ij,tjk->tik", init_rot, rel_rot).astype(np.float32)
        abs_rot_6d = (
            pt.matrix_to_rotation_6d(torch.from_numpy(abs_rot))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        abs_gripper = action_chunk[:, grip_slice].astype(np.float32)

        abs_components.extend([abs_pos, abs_rot_6d, abs_gripper])

    return np.concatenate(abs_components, axis=-1).astype(np.float32)


def run_inference(obs_dict: dict, policy, task_instruction: str, action_norm_stats, show_video: VideoStreamer) -> dict:

    images = obs_dict["images"]  # List[np.ndarray], (H, W, 3), uint8
    proprio = obs_dict["proprio"] 

    if not isinstance(images, dict) or len(images) < 2:
        raise ValueError("obs_dict['images'] must contain at least two images")
    #先left后right
    images = list(images.values())
    proprio = np.concatenate(list(obs_dict["proprio"].values()), axis=-1) #[20]
    show_video.update_frame(images[0], images[1])  # Example: show left and right wrist camera views

    example = {
        "image": images,
        "lang": task_instruction,
    }
    result = policy.predict_action(examples=example)

    normalized_action_chunk = parse_response(result)  # [T, 14]
    unorm_action_chunk = unnormalize_actions(normalized_action_chunk, action_norm_stats)

    final_actions = inverse_action(unorm_action_chunk, proprio)  # shape [T, action_dim]

    left_arm = final_actions[:, :9]            # [1,9]
    left_gripper = final_actions[:, 9:10]       # [1,1]
    right_arm = final_actions[:, 10:19]         # [1,9]
    right_gripper = final_actions[:, 19:20]     # [1,1]

    # 组装成你要的字典
    final_actions = {
        "left_arm": left_arm,
        "left_gripper": left_gripper,
        "right_arm": right_arm,
        "right_gripper": right_gripper
        
    }
    return {"actions": final_actions}

    


# ============================================================
# Main inference loop (dual-arm)
# ============================================================
def main():
    # ------ Configuration ------
    policy_host = "0.0.0.0"
    policy_port = 8888
    video_port = 7777
    task_instruction = "Two robotic arms are working together to perform a handover task."
    action_stats_path = "/mnt/home/yangzhibo/starVLA/examples/MyData/20260421/meta/relative_stats.json"
    ckpt_path = "/mnt/home/yangzhibo/starVLA/playground/Checkpoints/1011_starvla_qwenpi/checkpoints/steps_15000_pytorch_model.pt"
    use_bf16 = False

    # ------ ✅ Real code: load normalization statistics (dual-arm 14D) ------
    action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key="new_embodiment")
    print(f"Action dim: {len(action_norm_stats['q99'])}")  # should be 14
    print(f"Action q01: {action_norm_stats['q01']}")
    print(f"Action q99: {action_norm_stats['q99']}")

    # ------ ✅ Real code: load the Policy locally ------
    policy = build_local_policy(ckpt_path=ckpt_path, use_bf16=use_bf16)
    print(f"Loaded local policy from: {ckpt_path}")

    show_video = VideoStreamer(video_port)

    # 推理处理函数
    def handler(obs_dict: dict) -> dict:
        result = run_inference(obs_dict, policy, task_instruction, action_norm_stats, show_video)
        return result

    # 启动服务
    server = SimpleWebsocketServer(
        handler,
        host=policy_host,
        port=policy_port,
        metadata=None
        )

    try:
        server.serve_forever()
    finally:
        pass



if __name__ == "__main__":
    main()
