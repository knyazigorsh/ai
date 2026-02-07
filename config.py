from dataclasses import dataclass

@dataclass
class Config:
    # input
    source: str = "C:/Users/knyaz/Videos/20260203_1353_01kghj4ea0e858fecpz2m7c696.mp4"          # "0" = webcam, or path to video file
    out_path: str = ""         # e.g. "out.mp4" or "" to disable saving

    # detection
    det_conf: float = 0.5      # detector confidence threshold
    min_area: int = 36         # min bbox area (px^2) to ignore dust
    max_area: int = 99999999   # optional cap

    # tracking
    iou_match: float = 0.3     # IoU threshold for matching
    max_lost: int = 15         # frames to keep track without detections
    confirm_hits: int = 2      # need at least N hits in window
    confirm_window: int = 3    # window size for confirmation

    # visualization
    show: bool = True
    draw_fps: bool = True
