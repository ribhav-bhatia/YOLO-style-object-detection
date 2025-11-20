"""
Computer Vision for Robotic Systems — Real‑Time YOLO‑style Motion Detector
Tech: Python 3.9+, OpenCV, PyTorch (CUDA optional)

What this does
--------------
• Runs a YOLO‑style object detector in real time (webcam or video file).
• Computes a motion mask (frame differencing) and flags only *moving* detections.
• Designed as a lightweight inference pipeline you can drop into a robotics stack.
• Prints JSON detections per frame (for downstream consumers) and overlays results.

Install
-------
# Create/activate your venv first, then:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install opencv-python numpy

# Optional: ultralytics yolov5 via torch.hub (downloads on first run)
# (No extra pip needed; torch.hub will fetch from ultralytics/yolov5 automatically.)

Run
---
python realtime_cv_robotics.py --source 0  # webcam
python realtime_cv_robotics.py --source path/to/video.mp4
python realtime_cv_robotics.py --classes person car --conf 0.35 --move-thresh 20 --show 1

Outputs
-------
• Window with annotated frames (toggle by --show 1).  
• One JSON line per frame to stdout with timestamp, fps, and detections.

Integrating
-----------
• Pipe stdout into your robotics middleware (ROS2 node, ZeroMQ, etc.).  
• Use --publish-none 1 to suppress frames with no moving objects if desired.
"""
