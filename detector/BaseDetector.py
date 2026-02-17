from typing import List
import cv2
import cupy as cp
from .base import DetectorBase, Detection

def sigmoid(x):
    x = cp.clip(x, -50, 50)
    return 1.0 / (1.0 + cp.exp(-x))

def relu(x):
    return cp.maximum(0.0, x)

class BaseDetector(DetectorBase):
    def __init__(self, cell=64, stride=32, conf_thr=0.7, seed=0):
        self.cell = cell
        self.stride = stride
        self.conf_thr = conf_thr

        # пример: MLP вход = cell*cell (grayscale), скрытый 64, выход 1
        in_dim = cell * cell
        hidden = 64
        rng = cp.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden), dtype=cp.float32) * cp.sqrt(cp.float32(2.0 / in_dim)))
        self.b1 = cp.zeros((1, hidden), dtype=cp.float32)
        self.W2 = (rng.standard_normal((hidden, 1), dtype=cp.float32) * cp.sqrt(cp.float32(1.0 / hidden)))
        self.b2 = cp.zeros((1, 1), dtype=cp.float32)

    def _mlp(self, X):
        # X: (N, in_dim) float32 on GPU
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        P = sigmoid(Z2)  # (N,1)
        return P

    def infer(self, frame_bgr) -> List[Detection]:
        h, w = frame_bgr.shape[:2]

        # 1) grayscale + resize? (оставим как есть)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 2) нарезаем клетки
        patches = []
        coords = []
        for y in range(0, h - self.cell + 1, self.stride):
            for x in range(0, w - self.cell + 1, self.stride):
                patch = gray[y:y+self.cell, x:x+self.cell]
                patches.append(patch)
                coords.append((x, y))

        if not patches:
            return []

        # 3) batch на GPU: (N, cell*cell), нормализация 0..1
        X = cp.asarray(cp.stack([cp.asarray(p) for p in patches])).astype(cp.float32) / cp.float32(255.0)
        X = X.reshape(X.shape[0], -1)

        # 4) инференс
        P = self._mlp(X).reshape(-1)  # (N,)

        # 5) обратно в Detection
        dets: List[Detection] = []
        P_cpu = P.get()  # переносим только вероятности
        for (x, y), conf in zip(coords, P_cpu):
            if conf >= self.conf_thr:
                dets.append(Detection(x, y, x + self.cell, y + self.cell, conf=float(conf), cls=0))
                print("detections:", len(dets))
        return dets