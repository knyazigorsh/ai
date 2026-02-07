import os

USE_CUDA = os.getenv("USE_CUDA", "1") == "1"

def get_xp():
    if USE_CUDA:
        try:
            import cupy as cp
            dev = int(os.getenv("CUDA_DEVICE", "0"))
            cp.cuda.Device(dev).use()
            return cp
        except Exception:
            pass
    import numpy as np
    return np

xp = get_xp()
