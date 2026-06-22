from __future__ import annotations
import numpy as np

def is_symmetric(M, atol=1e-8):
    M = np.asarray(M)
    return np.allclose(M, M.T, atol=atol)

def assert_shape(arr, shape, name="array"):
    arr = np.asarray(arr)
    if arr.shape != tuple(shape):
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")

def regularize_psd(M, eps=1e-8):
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T) + eps * np.eye(M.shape[0])
