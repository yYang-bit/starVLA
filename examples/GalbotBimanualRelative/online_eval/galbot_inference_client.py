#!/usr/bin/env python3
"""
StarVLA Dual-Arm Inference Client Template
===========================================

Client 端职责:
  1. 读取机器人状态（双臂 abs eef pose）
  2. 采集双臂腕部相机图像
  3. (可选) 在 client 端 resize 图像到模型输入尺寸，减少网络传输
  4. 发送数据给 server 端
  5. 接收物理单位的 delta_action
  6. 执行机器人控制：abs[t+1] = abs[t] + delta[t]

通信协议:
  - HTTP POST to server /infer endpoint
  - Request: images (base64 or array) + abs_eef_pose (20D)
  - Response: delta_action (action_horizon, 20) in physical units

Usage:
  python galbot_inference_client.py \\
      --server_url http://192.168.1.100:5000 \\
      --control_freq 10 \\
      --action_horizon 30
"""

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
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        self.session = requests.Session()  # 复用 TCP 连接，减少延迟

    def health_check(self) -> bool:
        """Check if server is alive."""
        try:
            resp = self.session.get(f"{self.server_url}/health", timeout=self.timeout)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

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
        left_eef_pose: np.ndarray,
        right_eef_pose: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Send inference request to server.

        Args:
            left_wrist_img: [H, W, 3] uint8 RGB
            right_wrist_img: [H, W, 3] uint8 RGB
            left_eef_pose: [10] = [xyz(3), rot6d(6), gripper(1)]
            right_eef_pose: [10] = [xyz(3), rot6d(6), gripper(1)]

        Returns:
            delta_action: [action_horizon, 20] physical units
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
                "left": left_eef_pose.tolist(),
                "right": right_eef_pose.tolist(),
            },
        }

        # Send POST request
        t0 = time.time()
        resp = self.session.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=self.timeout,
        )
        roundtrip_ms = (time.time() - t0) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"Server returned {resp.status_code}: {resp.text}")

        result = resp.json()
        delta_action = np.array(result["delta_action"], dtype=np.float32)
        inference_time_ms = result["inference_time_ms"]

        logger.info(
            f"Inference: server={inference_time_ms:.1f}ms, roundtrip={roundtrip_ms:.1f}ms"
        )

        return delta_action, inference_time_ms


# ==========================================
# Robot Interface (Mock - replace with real control)
# ==========================================
class MockRobotInterface:
    """Mock robot interface for demonstration. Replace with your actual robot API."""

    def __init__(self):
        # Dummy state: 双臂 abs eef pose
        self.left_xyz = np.array([300.0, 0.0, 200.0], dtype=np.float32)  # mm
        self.left_rot = Rotation.identity()
        self.left_gripper = 0.0  # 0=closed, 1=open

        self.right_xyz = np.array([-300.0, 0.0, 200.0], dtype=np.float32)  # mm
        self.right_rot = Rotation.identity()
        self.right_gripper = 0.0

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get current abs eef pose.

        Returns:
            left_pose: [10] = [xyz(3), rot6d(6), gripper(1)]
            right_pose: [10] = [xyz(3), rot6d(6), gripper(1)]
        """
        left_rot6d = self._rotation_to_rot6d(self.left_rot)
        right_rot6d = self._rotation_to_rot6d(self.right_rot)

        left_pose = np.concatenate([self.left_xyz, left_rot6d, [self.left_gripper]])
        right_pose = np.concatenate([self.right_xyz, right_rot6d, [self.right_gripper]])

        return left_pose.astype(np.float32), right_pose.astype(np.float32)

    def capture_images(self, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture images from wrist cameras and resize.

        Args:
            image_size: (H, W) target size for model input

        Returns:
            left_img: [H, W, 3] uint8 RGB
            right_img: [H, W, 3] uint8 RGB
        """
        # TODO: Replace with real camera capture
        # Example using OpenCV:
        #   left_cap = cv2.VideoCapture(0)
        #   ret, left_frame = left_cap.read()
        #   left_img = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        #   left_img = cv2.resize(left_img, (image_size[1], image_size[0]))

        # Mock: generate random images
        left_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

        return left_img, right_img

    def execute_delta_action(self, delta_action: np.ndarray):
        """
        Execute one step of delta action.

        Args:
            delta_action: [20] = left [xyz(3), rot6d(6), gripper(1)] + right [xyz(3), rot6d(6), gripper(1)]
        """
        # Parse delta
        left_delta_xyz = delta_action[0:3]
        left_delta_rot6d = delta_action[3:9]
        left_delta_gripper = delta_action[9]

        right_delta_xyz = delta_action[10:13]
        right_delta_rot6d = delta_action[13:19]
        right_delta_gripper = delta_action[19]

        # Update abs state: abs[t+1] = abs[t] + delta[t]
        self.left_xyz += left_delta_xyz
        self.left_rot = self._apply_delta_rotation(self.left_rot, left_delta_rot6d)
        self.left_gripper = np.clip(self.left_gripper + left_delta_gripper, 0.0, 1.0)

        self.right_xyz += right_delta_xyz
        self.right_rot = self._apply_delta_rotation(self.right_rot, right_delta_rot6d)
        self.right_gripper = np.clip(self.right_gripper + right_delta_gripper, 0.0, 1.0)

        # TODO: Send commands to real robot
        logger.debug(f"Executed delta: L_xyz={left_delta_xyz}, R_xyz={right_delta_xyz}")

    def _rotation_to_rot6d(self, rot: Rotation) -> np.ndarray:
        """Convert scipy Rotation to rot6d (first two columns of rotation matrix)."""
        R = rot.as_matrix()  # (3, 3)
        return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)  # (6,)

    def _apply_delta_rotation(self, current_rot: Rotation, delta_rot6d: np.ndarray) -> Rotation:
        """
        Apply delta rotation (SO(3) relative rotation).

        R_new = R_current @ R_delta
        """
        # Convert delta_rot6d to rotation matrix
        from starVLA_repo_placeholder import rot6d_to_matrix  # TODO: import from your codebase

        R_delta = rot6d_to_matrix(delta_rot6d[None, :])[0]  # (3, 3)
        R_current = current_rot.as_matrix()
        R_new = R_current @ R_delta

        return Rotation.from_matrix(R_new)


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
    args = parser.parse_args()

    # Initialize client
    client = InferenceClient(
        server_url=args.server_url,
        timeout=5.0,
        use_jpeg_compression=not args.no_jpeg,
        jpeg_quality=args.jpeg_quality,
    )

    # Health check
    if not client.health_check():
        logger.error("Server health check failed. Is the server running?")
        return

    logger.info(f"Connected to server: {args.server_url}")

    # Initialize robot interface
    robot = MockRobotInterface()

    # Control loop parameters
    dt = 1.0 / args.control_freq
    action_buffer = None  # Buffer to store action chunk
    action_idx = 0  # Current index in action buffer

    logger.info(f"Starting control loop at {args.control_freq} Hz")

    try:
        for step in range(args.max_steps):
            t_start = time.time()

            # Step 1: Get current robot state
            left_pose, right_pose = robot.get_current_state()

            # Step 2: Capture images (resize at client to reduce transmission)
            left_img, right_img = robot.capture_images(tuple(args.image_size))

            # Step 3: Request new action chunk if buffer is empty or exhausted
            if action_buffer is None or action_idx >= len(action_buffer):
                logger.info(f"Step {step}: Requesting new action chunk from server")
                action_buffer, inference_time = client.infer(
                    left_img, right_img, left_pose, right_pose
                )
                action_idx = 0

            # Step 4: Execute one step from action buffer
            current_action = action_buffer[action_idx]
            robot.execute_delta_action(current_action)
            action_idx += 1

            logger.info(
                f"Step {step}: action_idx={action_idx}/{len(action_buffer)}, "
                f"L_xyz={left_pose[:3]}, R_xyz={right_pose[:3]}"
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

    logger.info("Control loop finished")


if __name__ == "__main__":
    main()
