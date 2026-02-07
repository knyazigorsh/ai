from dataclasses import dataclass, field
from typing import List, Tuple
from .iou import iou_xyxy

@dataclass
class Track:
    tid: int
    bbox: Tuple[float, float, float, float]  # xyxy
    conf: float
    age: int = 0          # total frames seen
    lost: int = 0         # frames since last matched
    hit_history: List[int] = field(default_factory=list)  # 1/0 per frame
    confirmed: bool = False

class SimpleTracker:
    def __init__(self, iou_match=0.3, max_lost=15, confirm_hits=2, confirm_window=3):
        self.iou_match = float(iou_match)
        self.max_lost = int(max_lost)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self._next_id = 1
        self.tracks: List[Track] = []

    def _is_confirmed(self, tr: Track) -> bool:
        if len(tr.hit_history) < self.confirm_window:
            return False
        return sum(tr.hit_history[-self.confirm_window:]) >= self.confirm_hits

    def update(self, dets_xyxy_conf: List[Tuple[Tuple[float,float,float,float], float]]) -> List[Track]:
        # mark all existing tracks as not matched this frame (hit=0 by default)
        for tr in self.tracks:
            tr.age += 1
            tr.lost += 1
            tr.hit_history.append(0)

        # greedy matching by IoU (simple but works as a starting point)
        unmatched_dets = list(range(len(dets_xyxy_conf)))
        for tr in self.tracks:
            best_j = -1
            best_iou = 0.0
            for j in unmatched_dets:
                bbox_j, conf_j = dets_xyxy_conf[j]
                v = iou_xyxy(tr.bbox, bbox_j)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j != -1 and best_iou >= self.iou_match:
                bbox_j, conf_j = dets_xyxy_conf[best_j]
                tr.bbox = bbox_j
                tr.conf = conf_j
                tr.lost = 0
                tr.hit_history[-1] = 1
                unmatched_dets.remove(best_j)

        # create new tracks for unmatched detections
        for j in unmatched_dets:
            bbox_j, conf_j = dets_xyxy_conf[j]
            tr = Track(
                tid=self._next_id,
                bbox=bbox_j,
                conf=conf_j,
                age=1,
                lost=0,
                hit_history=[1],
            )
            self._next_id += 1
            self.tracks.append(tr)

        # drop dead tracks
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        # attach confirmed flag dynamically (optional, for UI)
        for tr in self.tracks:
            tr.confirmed = self._is_confirmed(tr)  # type: ignore[attr-defined]
        return self.tracks
