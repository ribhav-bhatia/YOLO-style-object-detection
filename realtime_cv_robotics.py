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

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torchvision
import cv2

############################################################
# Utilities
############################################################

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad image while meeting stride-multiple constraints.
    Returns padded image, scale ratio, and padding (dw, dh).
    """
    shape = img.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([ (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1 ])


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    return torchvision.ops.nms(boxes, scores, iou_thres)

############################################################
# Motion detector (frame differencing)
############################################################
class MotionDetector:
    def __init__(self, alpha: float = 0.9, thresh: int = 25, min_area: int = 150):
        self.bg: Optional[np.ndarray] = None
        self.alpha = alpha
        self.thresh = thresh
        self.min_area = min_area

    def update(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.bg is None:
            self.bg = gray.astype(np.float32)
        cv2.accumulateWeighted(gray, self.bg, 1.0 - self.alpha)
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg))
        _, mask = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append(np.array([x, y, x + w, y + h]))
        return mask, boxes

    def box_is_moving(self, box_xyxy: np.ndarray, motion_boxes: List[np.ndarray]) -> bool:
        x1, y1, x2, y2 = box_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for mb in motion_boxes:
            mx1, my1, mx2, my2 = mb
            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                return True
        return False

############################################################
# Detector wrapper (YOLOv5 via torch.hub by default)
############################################################
class YOLODetector:
    def __init__(self, device: str = "cuda", conf_thres: float = 0.35, iou_thres: float = 0.45, half: bool = True, classes: Optional[List[str]] = None):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.names: List[str] = []
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.half = half and self.device.type == "cuda"
        self.class_filter = set(classes or [])
        self._load_model()

    def _load_model(self):
        # Load YOLOv5s from torch.hub (first run will download weights).
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape(False)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        # class names from repo
        self.names = self.model.names if hasattr(self.model, 'names') else [str(i) for i in range(80)]

    @torch.inference_mode()
    def infer(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], np.ndarray]:
        # Prepare
        img, r, (dw, dh) = letterbox(img_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float()  # CHW
        if self.half:
            tensor = tensor.half()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)

        # Model forward
        pred = self.model(tensor)[0]  # (N, 85) for yolov5: x1,y1,x2,y2,conf,cls_conf(s)

        # YOLOv5 returns (B, num, 85). We'll treat [:,:4], confidence at : , and cls index
        boxes = pred[..., :4]
        scores = pred[..., 4]
        cls_scores, cls_idx = pred[..., 5:].max(-1)
        scores = scores * cls_scores

        # Filter by conf
        keep = scores > self.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        cls_idx = cls_idx[keep]

        # NMS
        keep_idx = nms_pytorch(boxes, scores, self.iou_thres)
        boxes = boxes[keep_idx].detach().cpu().float().numpy()
        scores = scores[keep_idx].detach().cpu().float().numpy()
        cls_idx = cls_idx[keep_idx].detach().cpu().int().numpy()

        # Map back to original image scale
        if len(boxes):
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= r
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_bgr.shape[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_bgr.shape[0])
        det = np.concatenate([boxes, scores[:, None], cls_idx[:, None]], axis=1) if len(boxes) else np.zeros((0, 6), dtype=np.float32)
        return img, r, (dw, dh), det

    def class_name(self, idx: int) -> str:
        try:
            return self.names[idx]
        except Exception:
            return str(idx)

    def class_allowed(self, idx: int) -> bool:
        return (not self.class_filter) or (self.class_name(idx) in self.class_filter)

############################################################
# Main processing loop
############################################################
@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_name: str
    moving: bool

class VideoProcessor:
    def __init__(self, source: str | int, detector: YOLODetector, motion: MotionDetector, show: bool, publish_none: bool, max_fps: Optional[float], draw_motion: bool):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.detector = detector
        self.motion = motion
        self.show = show
        self.publish_none = publish_none
        self.max_fps = max_fps
        self.draw_motion = draw_motion

    def run(self):
        prev = time.time()
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # Limit FPS
            if self.max_fps:
                now = time.time()
                dt = now - prev
                min_dt = 1.0 / self.max_fps
                if dt < min_dt:
                    time.sleep(min_dt - dt)
                prev = time.time()

            t0 = time.time()
            motion_mask, motion_boxes = self.motion.update(frame)
            _, _, _, det = self.detector.infer(frame)

            detections: List[Detection] = []
            for x1, y1, x2, y2, conf, cls_idx in det:
                cls_idx = int(cls_idx)
                if not self.detector.class_allowed(cls_idx):
                    continue
                moving = self.motion.box_is_moving(np.array([x1, y1, x2, y2]), motion_boxes)
                detections.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(conf), self.detector.class_name(cls_idx), moving))

            # Visualization
            vis = frame.copy()
            if self.draw_motion:
                vis_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(vis, 1.0, vis_mask, 0.5, 0)

            for d in detections:
                color = (0, 255, 0) if d.moving else (128, 128, 128)
                x1, y1, x2, y2 = map(int, d.xyxy)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{d.cls_name} {d.conf:.2f}{' MOV' if d.moving else ''}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            fps = 1.0 / max(1e-6, (time.time() - t0))

            # Publish JSON line per frame
            payload = {
                "frame": frame_id,
                "ts": time.time(),
                "fps": round(fps, 2),
                "detections": [
                    {
                        "bbox_xyxy": [round(v, 2) for v in d.xyxy],
                        "conf": round(d.conf, 3),
                        "class": d.cls_name,
                        "moving": d.moving,
                    }
                    for d in detections if (self.publish_none or d.moving)
                ],
            }
            # Only emit frames if there is at least one moving detection unless publish_none
            if self.publish_none or any(d.moving for d in detections):
                sys.stdout.write(json.dumps(payload) + "\n")
                sys.stdout.flush()

            if self.show:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.imshow("CV Robotics — YOLO Motion", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

            frame_id += 1

        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()

############################################################
# CLI
############################################################

def parse_args():
    p = argparse.ArgumentParser(description="Real-time YOLO-style detector with motion gating for robotics")
    p.add_argument("--source", type=str, default="0", help="Video source (index or path)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--classes", nargs="*", default=None, help="Filter by class names (e.g., person car dog)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device preference")
    p.add_argument("--half", type=int, default=1, help="Use FP16 on CUDA (1/0)")
    p.add_argument("--show", type=int, default=1, help="Show annotated video (1/0)")
    p.add_argument("--publish-none", type=int, default=1, help="Publish frames even if no moving detections (1/0)")
    p.add_argument("--move-alpha", type=float, default=0.9, help="Motion EMA alpha (closer to 1 = slower bg update)")
    p.add_argument("--move-thresh", type=int, default=25, help="Motion pixel threshold (0–255)")
    p.add_argument("--move-min-area", type=int, default=150, help="Minimum contour area to treat as motion")
    p.add_argument("--draw-motion", type=int, default=0, help="Overlay motion mask (1/0)")
    p.add_argument("--max-fps", type=float, default=None, help="Cap processing FPS")
    return p.parse_args()


def main():
    args = parse_args()
    source: str | int = 0
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    detector = YOLODetector(device=args.device, conf_thres=args.conf, iou_thres=args.iou, half=bool(args.half), classes=args.classes)
    motion = MotionDetector(alpha=args.move_alpha, thresh=args.move_thresh, min_area=args.move_min_area)
    vp = VideoProcessor(source, detector, motion, show=bool(args.show), publish_none=bool(args.publish_none), max_fps=args.max_fps, draw_motion=bool(args.draw_motion))
    try:
        vp.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
