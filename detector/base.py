from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    # xyxy in pixels
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0  # one class: drone

class DetectorBase:
    def infer(self, frame_bgr) -> List[Detection]:
        raise NotImplementedError
