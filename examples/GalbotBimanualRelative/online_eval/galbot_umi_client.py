#!/usr/bin/env python3
"""Dual-arm HTTP infer client matched to the latest umi_client_dual control stack."""

from __future__ import annotations

import argparse
import base64
import copy
import importlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation


CURRENT_DIR = Path(__file__).resolve().parent
UMI_CLIENT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(UMI_CLIENT_ROOT))

GalbotControl = None
LoggerManager = None


def _load_local_args_class():
    args_path = UMI_CLIENT_ROOT / "args.py"
    spec = importlib.util.spec_from_file_location("umi_client_dual_local_args", str(args_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local args.py from {args_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["args"] = module
    return module.Args


def _load_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_local_namespace_package(package_name: str, package_path: Path) -> None:
    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module


_ensure_local_namespace_package("tool", UMI_CLIENT_ROOT / "tool")
_ensure_local_namespace_package("galbot_control", UMI_CLIENT_ROOT / "galbot_control")
_ensure_local_namespace_package("model_agent", UMI_CLIENT_ROOT / "model_agent")

Args = _load_local_args_class()
ShutdownTool = _load_local_module(
    "umi_client_dual_local_tool_shutdown",
    UMI_CLIENT_ROOT / "tool" / "tool_shutdown.py",
).ShutdownTool


def _default_host() -> str:
    raw_hosts = getattr(Args, "host", None)
    if isinstance(raw_hosts, list) and raw_hosts:
        return str(raw_hosts[0])
    return "127.0.0.1"


DEFAULT_SERVER_HOST = os.environ.get("OPENPI_UMI_SERVER_HOST", _default_host())
DEFAULT_SERVER_PORT = int(os.environ.get("OPENPI_UMI_SERVER_PORT", "5005"))
DEFAULT_PROMPT = os.environ.get("OPENPI_UMI_DEFAULT_PROMPT", "pick up the pen")
LOCAL_INIT_POSE_PATH = CURRENT_DIR / "init_pose.json"
HTTP_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("OPENPI_UMI_HTTP_CONNECT_TIMEOUT", "5"))
HTTP_READ_TIMEOUT_SECONDS = float(os.environ.get("OPENPI_UMI_HTTP_READ_TIMEOUT", "180"))
REPLAN_STEPS = int(os.environ.get("OPENPI_REPLAN_STEPS", "10"))
STATE_TIMEOUT_SECONDS = float(os.environ.get("OPENPI_STATE_TIMEOUT", "180"))
JPEG_QUALITY = int(os.environ.get("OPENPI_UMI_JPEG_QUALITY", "95"))


def _to_numpy_action_chunk(candidate: Any) -> np.ndarray | None:
    if candidate is None:
        return None
    actions = np.asarray(candidate, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        return actions[0]
    if actions.ndim == 2:
        return actions
    if actions.ndim == 1:
        return actions[None, :]
    return None


def _encode_image_to_base64(image_rgb: np.ndarray, jpeg_quality: int = JPEG_QUALITY) -> str:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _arm_state_to_pose8(arm_9d: np.ndarray, gripper_1d: np.ndarray) -> np.ndarray:
    arm_9d = np.asarray(arm_9d, dtype=np.float32).reshape(-1)
    gripper_1d = np.asarray(gripper_1d, dtype=np.float32).reshape(-1)
    pos = arm_9d[:3]
    x_axis = arm_9d[3:6]
    y_axis = arm_9d[6:9]

    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm < 1e-6 or y_norm < 1e-6:
        rot_mat = np.eye(3, dtype=np.float32)
    else:
        x_axis = x_axis / x_norm
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            rot_mat = np.eye(3, dtype=np.float32)
        else:
            y_axis = y_axis / y_norm
            z_axis = np.cross(x_axis, y_axis)
            z_norm = np.linalg.norm(z_axis)
            if z_norm < 1e-6:
                rot_mat = np.eye(3, dtype=np.float32)
            else:
                z_axis = z_axis / z_norm
                y_axis = np.cross(z_axis, x_axis)
                rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)

    quat = Rotation.from_matrix(rot_mat).as_quat().astype(np.float32)
    return np.concatenate([pos.astype(np.float32), quat, np.array([float(gripper_1d[0])], dtype=np.float32)], axis=0)


def _pose8_to_arm9(pose_8d: np.ndarray) -> np.ndarray:
    pose_8d = np.asarray(pose_8d, dtype=np.float32).reshape(-1)
    rot_mat = Rotation.from_quat(pose_8d[3:7]).as_matrix().astype(np.float32)
    return np.concatenate([pose_8d[:3], rot_mat[:, 0], rot_mat[:, 1]], axis=0)


def _apply_delta_action(current_pose_8d: np.ndarray, pred_action_7d: np.ndarray) -> np.ndarray:
    current_pose_8d = np.asarray(current_pose_8d, dtype=np.float32)
    pred_action_7d = np.asarray(pred_action_7d, dtype=np.float32)

    curr_pos = current_pose_8d[:3]
    curr_quat = current_pose_8d[3:7]
    curr_rot = Rotation.from_quat(curr_quat)

    delta_pos_world = curr_rot.apply(pred_action_7d[:3])
    next_pos = curr_pos + delta_pos_world
    next_rot = curr_rot * Rotation.from_rotvec(pred_action_7d[3:6])
    next_quat = next_rot.as_quat().astype(np.float32)
    next_gripper = np.array([float(pred_action_7d[6])], dtype=np.float32)
    return np.concatenate([next_pos.astype(np.float32), next_quat, next_gripper], axis=0)


def _rollout_target_poses(start_pose_8d: np.ndarray, action_chunk_7d: np.ndarray) -> np.ndarray:
    virtual_pose = np.asarray(start_pose_8d, dtype=np.float32)
    outputs = []
    for action in np.asarray(action_chunk_7d, dtype=np.float32):
        virtual_pose = _apply_delta_action(virtual_pose, action)
        outputs.append(virtual_pose.copy())
    return np.asarray(outputs, dtype=np.float32)


def _load_init_pose(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    init_pose = data.get("init_pose")
    if not isinstance(init_pose, list):
        raise ValueError(f"'init_pose' must be a list in {path}")
    if len(init_pose) != 23:
        raise ValueError(f"Expected 23 values in init_pose, got {len(init_pose)} from {path}")
    return [float(x) for x in init_pose]


def wait_for_robot_state_ready(galbot, shutdown_event: threading.Event, timeout_s: float, logger) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline and not shutdown_event.is_set():
        has_joint_sensor = galbot.galbot_interface._joint_sensor_vla is not None
        pose_len = len(galbot.galbot_interface.pose_buffer)
        tf_len = len(galbot.galbot_interface.T_buffer)
        if has_joint_sensor and pose_len >= 2 and tf_len >= 2:
            logger.info("robot state is ready")
            return
        logger.info(
            "waiting for robot state: joint_sensor=%s pose_buffer=%d tf_buffer=%d",
            has_joint_sensor,
            pose_len,
            tf_len,
        )
        time.sleep(1.0)
    raise RuntimeError(
        "Robot state is not ready yet after waiting. "
        f"joint_sensor={galbot.galbot_interface._joint_sensor_vla is not None}, "
        f"pose_buffer={len(galbot.galbot_interface.pose_buffer)}, "
        f"tf_buffer={len(galbot.galbot_interface.T_buffer)}"
    )


class RemoteUMIDualHttpAgent:
    def __init__(self, host: str, port: int, prompt: str):
        self.host = host
        self.port = int(port)
        self.prompt = prompt
        self.url = f"http://{self.host}:{self.port}/predict_action"
        self.health_url = f"http://{self.host}:{self.port}/health"
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def check_health(self) -> dict[str, Any]:
        response = self.session.get(
            self.health_url,
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response.json()

    def _build_payload(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("images", {})
        left_image = np.asarray(images["left_arm_camera"], dtype=np.uint8)
        right_image = np.asarray(images["right_arm_camera"], dtype=np.uint8)
        return {
            "examples": [
                {
                    "umi_left_color": _encode_image_to_base64(left_image),
                    "umi_right_color": _encode_image_to_base64(right_image),
                    "lang": self.prompt,
                }
            ]
        }

    def _extract_actions(self, result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = [
            result.get("actions"),
            result.get("data", {}).get("actions") if isinstance(result.get("data"), dict) else None,
            result.get("data", {}).get("unnormalized_actions") if isinstance(result.get("data"), dict) else None,
        ]
        for candidate in candidates:
            merged = _to_numpy_action_chunk(candidate)
            if merged is None:
                continue
            if merged.shape[-1] != 14:
                raise ValueError(f"Expected 14D dual-arm actions, got shape={merged.shape}")
            return merged, merged[:, :7], merged[:, 7:14]

        left = _to_numpy_action_chunk(result.get("left_actions"))
        right = _to_numpy_action_chunk(result.get("right_actions"))
        if left is None and isinstance(result.get("data"), dict):
            left = _to_numpy_action_chunk(result["data"].get("left_actions"))
        if right is None and isinstance(result.get("data"), dict):
            right = _to_numpy_action_chunk(result["data"].get("right_actions"))
        if left is None or right is None:
            raise ValueError(f"Server response does not contain dual-arm actions: {result}")
        if left.shape[0] != right.shape[0]:
            raise ValueError(f"Left/right action chunk length mismatch: {left.shape} vs {right.shape}")
        return np.concatenate([left, right], axis=-1), left, right

    def _format_actions_for_galbot(self, obs: dict[str, Any], left_chunk: np.ndarray, right_chunk: np.ndarray) -> dict[str, np.ndarray]:
        proprio = obs.get("proprio", {})
        left_pose = _arm_state_to_pose8(proprio["left_arm"][0], proprio["left_gripper"][0])
        right_pose = _arm_state_to_pose8(proprio["right_arm"][0], proprio["right_gripper"][0])

        left_targets = _rollout_target_poses(left_pose, left_chunk)
        right_targets = _rollout_target_poses(right_pose, right_chunk)

        return {
            "left_arm": np.asarray([_pose8_to_arm9(pose) for pose in left_targets], dtype=np.float32),
            "right_arm": np.asarray([_pose8_to_arm9(pose) for pose in right_targets], dtype=np.float32),
            "left_gripper": left_targets[:, 7:8].astype(np.float32),
            "right_gripper": right_targets[:, 7:8].astype(np.float32),
            "left_delta": np.asarray(left_chunk, dtype=np.float32),
            "right_delta": np.asarray(right_chunk, dtype=np.float32),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            self.url,
            json=self._build_payload(obs),
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") not in (None, "ok"):
            raise RuntimeError(result.get("error", f"Unexpected server response: {result}"))

        action_chunk, left_chunk, right_chunk = self._extract_actions(result)
        return {
            "status": result.get("status", "ok"),
            "actions": self._format_actions_for_galbot(obs, left_chunk, right_chunk),
            "data": {
                "unnormalized_actions": [action_chunk.tolist()],
                "inference_time": result.get("data", {}).get("inference_time", result.get("inference_time")),
            },
            "raw_action_chunk": action_chunk,
            "raw_response": result,
        }


class DualArmGalbotVLA:
    def __init__(self, args: Args, prompt: str, server_host: str, server_port: int, replan_steps: int, state_timeout: float):
        self.args = copy.deepcopy(args)
        self.prompt = prompt
        self.replan_steps = int(replan_steps)
        self.state_timeout = float(state_timeout)

        self.model_agent = RemoteUMIDualHttpAgent(server_host, server_port, prompt)
        self.galbot = GalbotControl(args)
        self.flag_has_moved_to_init_pose = False
        self.flag_model_first_infer = True
        self.shutdown_event = threading.Event()
        self.vla_model_error = False
        self.logger = LoggerManager.get_logger()

    def move_to_task_init_pose(self) -> None:
        init_pose = _load_init_pose(LOCAL_INIT_POSE_PATH)
        self.galbot.init_pose = init_pose

        wait_for_robot_state_ready(self.galbot, self.shutdown_event, self.state_timeout, self.logger)

        self.galbot.galbot_interface.clear_task_pose()
        time.sleep(0.5)
        self.logger.info("start task-specific init pose: %s", LOCAL_INIT_POSE_PATH)
        self.galbot.set_wholebody_angle_asynchronous(
            np.array([2] + init_pose + self.galbot.galbot_interface.chassis_pos).reshape(-1, 1),
            0.03,
            4,
        )
        self.logger.info("finish task-specific init pose")

        pose_wholebody = self.galbot.galbot_interface.pose_buffer[1]
        transform = self.galbot.galbot_interface.T_buffer[1]
        self.galbot.T_end_init = copy.deepcopy(transform)
        for _ in range(200):
            self.galbot.galbot_interface.pose_buffer.appendleft(pose_wholebody)
            self.galbot.galbot_interface.T_buffer.appendleft(transform)

    def _truncate_response_for_replan(self, response: dict[str, Any]) -> dict[str, Any]:
        if self.replan_steps <= 0:
            return response

        effective_steps = max(2, int(self.replan_steps))
        truncated = copy.deepcopy(response)
        actions = truncated.get("actions", {})
        if isinstance(actions, dict):
            for key, value in list(actions.items()):
                arr = np.asarray(value)
                if arr.ndim >= 1:
                    actions[key] = arr[:effective_steps]

        raw_action_chunk = truncated.get("raw_action_chunk")
        if raw_action_chunk is not None:
            truncated["raw_action_chunk"] = np.asarray(raw_action_chunk)[:effective_steps]

        data = truncated.get("data")
        if isinstance(data, dict):
            raw_unnorm = data.get("unnormalized_actions")
            if isinstance(raw_unnorm, list) and raw_unnorm:
                raw_chunk = np.asarray(raw_unnorm[0])
                data["unnormalized_actions"] = [raw_chunk[:effective_steps].tolist()]

        return truncated

    def blocking_mode(self) -> None:
        while not self.shutdown_event.is_set():
            if not self.flag_has_moved_to_init_pose:
                self.move_to_task_init_pose()
                self.flag_has_moved_to_init_pose = True

            obs = self.galbot.get_obs_wholebody_compressed(self.flag_model_first_infer, 0)
            if self.flag_model_first_infer:
                self.flag_model_first_infer = False

            try:
                response = self.model_agent.infer(obs)
                if self.replan_steps > 0:
                    response = self._truncate_response_for_replan(response)
            except Exception as exc:
                error_msg = f"vla model error : {type(exc).__name__}: {exc}"
                self.logger.error(error_msg)
                self.galbot.error_imformation = self.galbot.error_imformation + error_msg + ", "
                self.vla_model_error = True
                self.logger.error(self.galbot.error_imformation)

            if self.shutdown_event.is_set():
                break
            if not self.vla_model_error:
                self.galbot.set_wholebody_command_dual(obs, response)

            if self.galbot.error_imformation != "":
                self.logger.error(self.galbot.error_imformation)
                self.shutdown_event.set()
                self.galbot.shutdown()
                break

    def run(self) -> tuple[bool, str]:
        self.logger.info("in_Vla")
        self.galbot.shutdown_event.clear()
        self.galbot.error_imformation = ""
        self.shutdown_event.clear()
        self.vla_model_error = False
        self.flag_has_moved_to_init_pose = False
        self.flag_model_first_infer = True

        health = self.model_agent.check_health()
        self.logger.info("remote server health: %s", health)
        self.blocking_mode()

        self.logger.info("out_Vla")
        if self.galbot.error_imformation == "":
            return True, "success"
        return False, self.galbot.error_imformation

    def shutdown(self) -> None:
        self.shutdown_event.set()
        self.model_agent.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-robot dual-arm infer client for remote UMI HTTP server.")
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST, help="Remote UMI server host.")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT, help="Remote UMI server HTTP port.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Task prompt sent to the remote server.")
    parser.add_argument("--replan-steps", type=int, default=REPLAN_STEPS, help="Execute N steps before replanning.")
    parser.add_argument("--startup-sleep", type=float, default=3.0, help="Sleep before entering the control loop.")
    parser.add_argument("--state-timeout", type=float, default=STATE_TIMEOUT_SECONDS, help="Maximum seconds to wait for robot state.")
    return parser.parse_args()


def main() -> int:
    global GalbotControl, LoggerManager
    cli_args = parse_args()

    if GalbotControl is None:
        GalbotControl = importlib.import_module("galbot_control.galbot_control").GalbotControl
    if LoggerManager is None:
        LoggerManager = importlib.import_module("tool.logger").LoggerManager

    runtime_args = Args()
    runtime_args.host = [cli_args.server_host]
    runtime_args.port = [int(cli_args.server_port)]

    galbotvla = DualArmGalbotVLA(
        args=runtime_args,
        prompt=cli_args.prompt,
        server_host=cli_args.server_host,
        server_port=int(cli_args.server_port),
        replan_steps=int(cli_args.replan_steps),
        state_timeout=float(cli_args.state_timeout),
    )

    tool_shutdown = ShutdownTool()
    tool_shutdown.on_shutdown(galbotvla.shutdown)
    tool_shutdown.on_shutdown(galbotvla.galbot.shutdown)

    time.sleep(float(cli_args.startup_sleep))
    success, message = galbotvla.run()
    if success:
        return 0
    print(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
