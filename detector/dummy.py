from typing import List
import cv2
from .base import DetectorBase, Detection

class DummyDetector(DetectorBase):
    def infer(self, frame_bgr) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        # bbox in center, size depends on image
        bw, bh = int(w * 0.08), int(h * 0.08)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh
        return [Detection(x1, y1, x2, y2, conf=0.9, cls=0)]
