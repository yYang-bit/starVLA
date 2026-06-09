#!/usr/bin/env python3
"""
StarVLA Dual-Arm Inference Client Template
===========================================

Client 端职责:
  1. 读取机器人状态（双臂 abs eef pose: xyz 米 + 四元数）
  2. 采集双臂腕部相机图像
  3. (可选) 在 client 端 resize 图像到模型输入尺寸，减少网络传输
  4. 发送数据给 server 端
  5. 接收物理单位的 delta_action (xyz 米 + 3x3 rotation matrix)
  6. 执行机器人控制：
     - delta_xyz_world = R_current @ delta_xyz_local
     - xyz_target = xyz_current + delta_xyz_world
     - R_target = R_current @ delta_rotation
     - quat_target = Rotation.from_matrix(R_target).as_quat()

Units and Conventions:
  - xyz: meters (m)
  - rotation: quaternion [x, y, z, w] (scipy convention)
  - delta xyz: EEF local frame, converted to world before publishing WBC targets
  - gripper: binary {0, 1}

Communication Protocol:
  - HTTP POST to server /infer endpoint
  - Request: images (base64 JPEG) + abs_eef_pose (xyz + quat)
  - Response: delta_action list (xyz + 3x3 rotation matrix + gripper)

Usage:
  python galbot_client.py \\
      --server_url http://192.168.1.100:5000 \\
      --control_freq 10 \\
      --action_horizon 30
"""

from __future__ import annotations

# ==========================================
# Configuration (modify these for your setup)
# ==========================================
DEFAULT_SERVER_URL = "http://localhost:5000"
DEFAULT_CONTROL_FREQ = 10  # Hz，机器人控制频率
DEFAULT_ACTION_HORIZON = 30  # 模型输出的 action chunk 长度
DEFAULT_IMAGE_SIZE = (224, 224)  # 模型输入图像尺寸 (H, W)
DEFAULT_USE_JPEG_COMPRESSION = True  # 是否用 JPEG 压缩图像再发送（减少传输量）
DEFAULT_JPEG_QUALITY = 85  # JPEG 压缩质量 (0-100)
# ==========================================

import argparse
import base64
import copy
import importlib
import importlib.util
import io
import json
import logging
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


CURRENT_DIR = Path(__file__).resolve().parent
UMI_CLIENT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(UMI_CLIENT_ROOT))


def _ensure_local_namespace_package(package_name: str, package_path: Path) -> None:
    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module


def _install_image_process_shim_if_missing() -> None:
    """Allow importing GalbotControl in envs without the compiled ImageProcess extension."""
    if "tool.ImageProcess" in sys.modules:
        return
    try:
        importlib.import_module("tool.ImageProcess")
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("tool.ImageProcess")

    class _ImageProcessShim:
        def __init__(self, *args, **kwargs):
            pass

        def numpy2D_to_str(self, arr, prec: int = 5) -> str:
            return str(np.asarray(arr))

    shim.ImageProcess = _ImageProcessShim
    sys.modules["tool.ImageProcess"] = shim


def _load_local_args_class():
    args_path = UMI_CLIENT_ROOT / "args.py"
    spec = importlib.util.spec_from_file_location("starvla_robot_client_args", str(args_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local args.py from {args_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["args"] = module
    return module.Args


_ensure_local_namespace_package("tool", UMI_CLIENT_ROOT / "tool")
_ensure_local_namespace_package("galbot_control", UMI_CLIENT_ROOT / "galbot_control")
_ensure_local_namespace_package("model_agent", UMI_CLIENT_ROOT / "model_agent")
_install_image_process_shim_if_missing()

Args = _load_local_args_class()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def wait_for_robot_state_ready(galbot, shutdown_event: threading.Event, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline and not shutdown_event.is_set():
        has_joint_sensor = galbot.galbot_interface._joint_sensor_vla is not None
        pose_len = len(galbot.galbot_interface.pose_buffer)
        tf_len = len(galbot.galbot_interface.T_buffer)
        has_valid_tf = False
        if tf_len >= 2:
            transform_now = galbot.galbot_interface.T_buffer[1]
            has_valid_tf = len(transform_now) >= 3 and all(len(pose) == 7 for pose in transform_now[:3])
        has_left_image = galbot.galbot_interface._image_hand_left is not None
        has_right_image = galbot.galbot_interface._image_hand_right is not None
        if has_joint_sensor and pose_len >= 2 and has_valid_tf and has_left_image and has_right_image:
            logger.info("Robot state is ready")
            return
        logger.info(
            "Waiting for robot state: joint_sensor=%s pose_buffer=%d tf_buffer=%d valid_tf=%s left_img=%s right_img=%s",
            has_joint_sensor,
            pose_len,
            tf_len,
            has_valid_tf,
            has_left_image,
            has_right_image,
        )
        time.sleep(1.0)
    raise RuntimeError(
        "Robot state is not ready after waiting. "
        f"joint_sensor={galbot.galbot_interface._joint_sensor_vla is not None}, "
        f"pose_buffer={len(galbot.galbot_interface.pose_buffer)}, "
        f"tf_buffer={len(galbot.galbot_interface.T_buffer)}, "
        f"left_image={galbot.galbot_interface._image_hand_left is not None}, "
        f"right_image={galbot.galbot_interface._image_hand_right is not None}"
    )


# ==========================================
# Communication Module
# ==========================================
class InferenceClient:
    """HTTP client for communicating with StarVLA inference server."""

    def __init__(
        self,
        server_url: str,
        timeout: float = 5.0,
        use_jpeg_compression: bool = True,
        jpeg_quality: int = 85,
    ):
        """
        Args:
            server_url: Server base URL (e.g., "http://192.168.1.100:5000")
            timeout: HTTP request timeout in seconds
            use_jpeg_compression: Whether to compress images with JPEG before sending
            jpeg_quality: JPEG compression quality (0-100, higher = better quality)
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.use_jpeg_compression = use_jpeg_compression
        self.jpeg_quality = jpeg_quality

    def health_check(self) -> bool:
        """Check if server is alive."""
        try:
            result = self._get_json("/health")
            return result.get("status") == "ok"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _get_json(self, endpoint: str) -> dict:
        request = urllib.request.Request(
            f"{self.server_url}{endpoint}",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{endpoint}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Server returned {exc.code}: {error_body}") from exc

    def encode_image(self, img: np.ndarray) -> str:
        """
        Encode numpy image to base64 string.

        Args:
            img: [H, W, 3] uint8 RGB image

        Returns:
            base64 encoded string (JPEG compressed if enabled)
        """
        if self.use_jpeg_compression:
            # Convert to PIL and save as JPEG in memory
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            img_bytes = buffer.getvalue()
        else:
            # Convert to PNG (lossless but larger)
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    def infer(
        self,
        left_wrist_img: np.ndarray,
        right_wrist_img: np.ndarray,
        left_xyz: np.ndarray,
        left_quat: np.ndarray,
        right_xyz: np.ndarray,
        right_quat: np.ndarray,
    ) -> tuple[list[dict], float]:
        """
        Send inference request to server.

        Args:
            left_wrist_img: [H, W, 3] uint8 RGB
            right_wrist_img: [H, W, 3] uint8 RGB
            left_xyz: [3] meters
            left_quat: [4] quaternion [x, y, z, w]
            right_xyz: [3] meters
            right_quat: [4] quaternion [x, y, z, w]

        Returns:
            delta_action: list of dicts, each with structure:
                {
                  "left": {"xyz": [3], "rotation": [[3,3]], "gripper": float},
                  "right": {"xyz": [3], "rotation": [[3,3]], "gripper": float}
                }
            inference_time_ms: server-side inference time
        """
        # Encode images
        left_b64 = self.encode_image(left_wrist_img)
        right_b64 = self.encode_image(right_wrist_img)

        # Build request payload
        payload = {
            "images": {
                "left_wrist": left_b64,
                "right_wrist": right_b64,
            },
            "abs_eef_pose": {
                "left": {
                    "xyz": left_xyz.tolist(),
                    "quat": left_quat.tolist(),
                },
                "right": {
                    "xyz": right_xyz.tolist(),
                    "quat": right_quat.tolist(),
                },
            },
        }

        # Send POST request
        t0 = time.time()
        result = self._post_json("/infer", payload)
        roundtrip_ms = (time.time() - t0) * 1000

        delta_action = result["delta_action"]
        inference_time_ms = result["inference_time_ms"]

        logger.info(
            f"Inference: server={inference_time_ms:.1f}ms, roundtrip={roundtrip_ms:.1f}ms"
        )

        return delta_action, inference_time_ms


# ==========================================
# Robot Interface
# ==========================================
class GalbotRobotInterface:
    """Robot interface backed by GalbotControl and the WBC EEF task interface."""

    def __init__(
        self,
        image_size: tuple[int, int],
        startup_sleep: float,
        state_timeout: float,
        gripper_closed: float,
        gripper_open: float,
    ):
        self.image_size = tuple(image_size)
        self.gripper_closed = float(gripper_closed)
        self.gripper_open = float(gripper_open)
        self.shutdown_event = threading.Event()
        self.args = Args()
        from galbot_control.galbot_control import GalbotControl

        self.galbot = GalbotControl(copy.deepcopy(self.args))
        self.left_target_xyz: np.ndarray | None = None
        self.left_target_quat: np.ndarray | None = None
        self.right_target_xyz: np.ndarray | None = None
        self.right_target_quat: np.ndarray | None = None

        time.sleep(float(startup_sleep))
        wait_for_robot_state_ready(self.galbot, self.shutdown_event, state_timeout)

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current abs eef pose.

        Returns:
            left_xyz: [3] meters
            left_quat: [4] quaternion [x, y, z, w]
            right_xyz: [3] meters
            right_quat: [4] quaternion [x, y, z, w]
        """
        transform_now = self.galbot.galbot_interface.T_buffer[1]
        left_pose7d = np.asarray(transform_now[0], dtype=np.float32).reshape(7)
        right_pose7d = np.asarray(transform_now[1], dtype=np.float32).reshape(7)
        return (
            left_pose7d[:3].copy(),
            left_pose7d[3:7].copy(),
            right_pose7d[:3].copy(),
            right_pose7d[3:7].copy(),
        )

    def capture_images(self, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture images from wrist cameras and resize.

        Args:
            image_size: (H, W) target size for model input

        Returns:
            left_img: [H, W, 3] uint8 RGB
            right_img: [H, W, 3] uint8 RGB
        """
        return (
            self._get_wrist_image_rgb("hand_left", image_size),
            self._get_wrist_image_rgb("hand_right", image_size),
        )

    def reset_action_anchor(
        self,
        left_xyz: np.ndarray,
        left_quat: np.ndarray,
        right_xyz: np.ndarray,
        right_quat: np.ndarray,
    ) -> None:
        """Start a new action chunk from the robot state used for inference."""
        self.left_target_xyz = np.asarray(left_xyz, dtype=np.float64).reshape(3).copy()
        self.left_target_quat = np.asarray(left_quat, dtype=np.float64).reshape(4).copy()
        self.right_target_xyz = np.asarray(right_xyz, dtype=np.float64).reshape(3).copy()
        self.right_target_quat = np.asarray(right_quat, dtype=np.float64).reshape(4).copy()

    def execute_delta_action(self, delta: dict):
        """
        Execute one step of delta action.

        Args:
            delta: dict with structure:
                {
                  "left": {"xyz": [3], "rotation": [[3,3]], "gripper": float},
                  "right": {"xyz": [3], "rotation": [[3,3]], "gripper": float}
                }
        """
        if self.left_target_xyz is None or self.left_target_quat is None:
            left_xyz, left_quat, right_xyz, right_quat = self.get_current_state()
            self.reset_action_anchor(left_xyz, left_quat, right_xyz, right_quat)

        left_delta_xyz = np.array(delta["left"]["xyz"], dtype=np.float32)
        left_delta_R = np.array(delta["left"]["rotation"], dtype=np.float32)  # (3, 3)
        left_delta_gripper = delta["left"]["gripper"]

        right_delta_xyz = np.array(delta["right"]["xyz"], dtype=np.float32)
        right_delta_R = np.array(delta["right"]["rotation"], dtype=np.float32)  # (3, 3)
        right_delta_gripper = delta["right"]["gripper"]

        R_left_current = Rotation.from_quat(self.left_target_quat).as_matrix()
        self.left_target_xyz = self.left_target_xyz + R_left_current @ left_delta_xyz.astype(np.float64)
        R_left_new = R_left_current @ left_delta_R
        self.left_target_quat = Rotation.from_matrix(R_left_new).as_quat().astype(np.float64)

        R_right_current = Rotation.from_quat(self.right_target_quat).as_matrix()
        self.right_target_xyz = self.right_target_xyz + R_right_current @ right_delta_xyz.astype(np.float64)
        R_right_new = R_right_current @ right_delta_R
        self.right_target_quat = Rotation.from_matrix(R_right_new).as_quat().astype(np.float64)

        head_pose = np.asarray(self.galbot.galbot_interface.T_buffer[1][2], dtype=np.float64).reshape(7)
        self.galbot.galbot_interface.pub_task_pose(
            [
                "left_arm_end_effector_mount_link",
                "right_arm_end_effector_mount_link",
                "head_base_link",
            ],
            [self.left_target_xyz, self.right_target_xyz, head_pose[:3]],
            [self.left_target_quat, self.right_target_quat, head_pose[3:7]],
        )
        self.galbot.galbot_interface.pub_gripper_command(
            "left_gripper",
            self._binary_gripper_to_command(left_delta_gripper),
            100,
        )
        self.galbot.galbot_interface.pub_gripper_command(
            "right_gripper",
            self._binary_gripper_to_command(right_delta_gripper),
            100,
        )

        logger.debug(
            "Executed target: L_xyz=%s R_xyz=%s",
            np.array2string(self.left_target_xyz, precision=4, suppress_small=True),
            np.array2string(self.right_target_xyz, precision=4, suppress_small=True),
        )

    def shutdown(self) -> None:
        self.shutdown_event.set()
        try:
            self.galbot.galbot_interface.clear_task_pose()
        except Exception:
            pass
        try:
            self.galbot.shutdown()
        except Exception:
            pass

    def _binary_gripper_to_command(self, value: float) -> float:
        return self.gripper_open if float(value) >= 0.5 else self.gripper_closed

    def _get_wrist_image_rgb(self, target: str, image_size: tuple[int, int]) -> np.ndarray:
        image_bytes, error = self.galbot.galbot_interface.get_image(target)
        if image_bytes is None:
            raise RuntimeError(error or f"{target} image is None")
        arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to decode {target} image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(image_rgb, (int(image_size[1]), int(image_size[0])))


# ==========================================
# Main Control Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="StarVLA Dual-Arm Inference Client")
    parser.add_argument("--server_url", type=str, default=DEFAULT_SERVER_URL, help="Inference server URL")
    parser.add_argument("--control_freq", type=float, default=DEFAULT_CONTROL_FREQ, help="Control frequency (Hz)")
    parser.add_argument("--action_horizon", type=int, default=DEFAULT_ACTION_HORIZON, help="Action chunk size")
    parser.add_argument("--image_size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, help="Image size (H W)")
    parser.add_argument("--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG compression quality")
    parser.add_argument("--no_jpeg", action="store_true", help="Disable JPEG compression (use PNG)")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum control steps (for testing)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--startup_sleep", type=float, default=3.0, help="Sleep before waiting for robot state")
    parser.add_argument("--state_timeout", type=float, default=180.0, help="Timeout for robot state readiness")
    parser.add_argument("--gripper_closed", type=float, default=50.0, help="Robot gripper command for binary 0")
    parser.add_argument("--gripper_open", type=float, default=120.0, help="Robot gripper command for binary 1")
    args = parser.parse_args()

    # Initialize client
    client = InferenceClient(
        server_url=args.server_url,
        timeout=args.timeout,
        use_jpeg_compression=not args.no_jpeg,
        jpeg_quality=args.jpeg_quality,
    )

    # Health check
    if not client.health_check():
        logger.error("Server health check failed. Is the server running?")
        return

    logger.info(f"Connected to server: {args.server_url}")

    robot = GalbotRobotInterface(
        image_size=tuple(args.image_size),
        startup_sleep=args.startup_sleep,
        state_timeout=args.state_timeout,
        gripper_closed=args.gripper_closed,
        gripper_open=args.gripper_open,
    )

    # Control loop parameters
    dt = 1.0 / args.control_freq
    action_buffer = None  # Buffer to store action chunk
    action_idx = 0  # Current index in action buffer

    logger.info(f"Starting control loop at {args.control_freq} Hz")

    try:
        for step in range(args.max_steps):
            t_start = time.time()

            # Step 1: Get current robot state (xyz in meters, quat [x,y,z,w])
            left_xyz, left_quat, right_xyz, right_quat = robot.get_current_state()

            # Step 2: Capture images (resize at client to reduce transmission)
            left_img, right_img = robot.capture_images(tuple(args.image_size))

            # Step 3: Request new action chunk if buffer is empty or exhausted
            if action_buffer is None or action_idx >= len(action_buffer):
                logger.info(f"Step {step}: Requesting new action chunk from server")
                action_buffer, inference_time = client.infer(
                    left_img, right_img,
                    left_xyz, left_quat,
                    right_xyz, right_quat,
                )
                robot.reset_action_anchor(left_xyz, left_quat, right_xyz, right_quat)
                action_idx = 0

            # Step 4: Execute one step from action buffer
            current_action = action_buffer[action_idx]
            robot.execute_delta_action(current_action)
            action_idx += 1

            logger.info(
                f"Step {step}: action_idx={action_idx}/{len(action_buffer)}, "
                f"L_xyz={left_xyz}, R_xyz={right_xyz}"
            )

            # Step 5: Sleep to maintain control frequency
            elapsed = time.time() - t_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Step {step}: Control loop exceeded target dt ({elapsed:.3f}s > {dt:.3f}s)")

    except KeyboardInterrupt:
        logger.info("Control loop interrupted by user")
    except Exception as e:
        logger.exception(f"Error in control loop: {e}")
    finally:
        robot.shutdown()

    logger.info("Control loop finished")


if __name__ == "__main__":
    main()
