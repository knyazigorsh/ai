import cv2

def draw_tracks(frame, tracks, color=(0, 255, 0)):
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox)
        conf = getattr(tr, "conf", 0.0)
        confirmed = getattr(tr, "confirmed", False)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tr.tid} {conf:.2f}" + (" OK" if confirmed else "")
        cv2.putText(frame, label, (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame
