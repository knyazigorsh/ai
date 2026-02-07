import time
import cv2

from config import Config
from detector import DummyDetector
from tracker import SimpleTracker
from utils.draw import draw_tracks

def open_source(src: str):
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)

def main():
    cfg = Config()

    cap = open_source(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {cfg.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_in = fps_in if fps_in and fps_in > 1 else 30.0

    writer = None
    if cfg.out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.out_path, fourcc, fps_in, (w, h))

    detector = DummyDetector()
    tracker = SimpleTracker(
        iou_match=cfg.iou_match,
        max_lost=cfg.max_lost,
        confirm_hits=cfg.confirm_hits,
        confirm_window=cfg.confirm_window,
    )

    t_prev = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- detect ---
        dets = detector.infer(frame)

        # filter by conf + area
        det_list = []
        for d in dets:
            if d.conf < cfg.det_conf:
                continue
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < cfg.min_area or area > cfg.max_area:
                continue
            det_list.append(((x1, y1, x2, y2), d.conf))

        # --- track ---
        tracks = tracker.update(det_list)

        # --- draw ---
        draw_tracks(frame, tracks)

        if cfg.draw_fps:
            t_now = time.perf_counter()
            dt = max(1e-6, t_now - t_prev)
            t_prev = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        if cfg.show:
            cv2.imshow("drone-tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
