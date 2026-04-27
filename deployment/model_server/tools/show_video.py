import threading
import time
import cv2
from flask import Flask, Response
import logging
import numpy as np

class VideoStreamer:
    """
    轻量级的后台视频流服务器。
    使用方法:
    1. streamer = VideoStreamer(port=5000)
    2. streamer.update_frame(cv2_image_bgr)
    """
    def __init__(self, port=5000):
        self.port = port
        self.latest_frame = None
        self.app = Flask(__name__)
        
        # 关闭 Flask 默认的烦人日志输出
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # 注册路由
        self.app.add_url_rule('/video', 'video', self.video_route)
        
        # 启动后台守护线程
        self.thread = threading.Thread(target=self._run_flask, daemon=True)
        self.thread.start()
        print(f"🎥 [VideoStreamer] 后台视频流已启动！监听端口: {self.port}")

    #for single frame
    # def update_frame(self, frame):
    #     """
    #     更新要在网页上显示的最新一帧图像。
    #     参数 frame: OpenCV 格式的图像 (BGR)
    #     """
    #     if frame is not None:
    #         # 必须用 copy()，防止在编码时原图被其他线程修改
    #         self.latest_frame = frame.copy()

    def update_frame(self, left_frame, right_frame):
        """
        更新左右双画面
        :param left_frame: 左侧OpenCV图像
        :param right_frame: 右侧OpenCV图像
        """
        if left_frame is not None and right_frame is not None:
            # 自动缩放到相同高度 + 左右拼接
            h = min(left_frame.shape[0], right_frame.shape[0])
            left = cv2.resize(left_frame, (int(left_frame.shape[1]*h/left_frame.shape[0]), h))
            right = cv2.resize(right_frame, (int(right_frame.shape[1]*h/right_frame.shape[0]), h))
            # RGB → BGR（OpenCV要求）
            left_frame_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            right_frame_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            # 水平拼接
            combined = np.hstack([left_frame_bgr, right_frame_bgr])
            self.latest_frame = combined.copy()

    def _generate_frames(self):
        while True:
            if self.latest_frame is None:
                time.sleep(0.05)
                continue
            
            # 将 OpenCV 图像编码为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', self.latest_frame)
            if not ret:
                time.sleep(0.05)
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 控制最大帧率，避免过度消耗 CPU (0.03 秒大约 30 帧)
            time.sleep(0.03)

    def video_route(self):
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _run_flask(self):
        # 禁用 debug 和 reloader，防止多线程冲突引发严重报错
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)