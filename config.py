from dataclasses import dataclass

@dataclass
class Config:
    
    source: str = "C:/Users/knyaz/Videos/20260203_1353_01kghj4ea0e858fecpz2m7c696.mp4"
    out_path: str = ""
    
    det_conf: float = 0.5
    min_area: int = 36
    max_area: int = 99999999
    
    iou_match: float = 0.3
    max_lost: int = 15
    confirm_hits: int = 2
    confirm_window: int = 3

    show: bool = True
    draw_fps: bool = True
