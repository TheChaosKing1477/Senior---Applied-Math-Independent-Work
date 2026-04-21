from __future__ import annotations
import numpy as np
from scipy.ndimage import percentile_filter

def interpolate_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    if good.sum() == 1:
        x[~good] = x[good][0]
        return x
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x

def estimate_bleach_baseline(x: np.ndarray, window: int=301, percentile: float=15.0) -> np.ndarray:
    x = interpolate_nans(x)
    window = max(5, int(window))
    if window % 2 == 0:
        window += 1
    return percentile_filter(x, percentile=percentile, size=window, mode="nearest").astype(float)

def correct_photobleaching(x: np.ndarray, window: int=301, percentile: float=15.0) -> np.ndarray:
    return interpolate_nans(x) - estimate_bleach_baseline(x, window=window, percentile=percentile)

def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x-med))
    scale = 1.4826*mad if mad > 1e-10 else np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return (x-med)/scale

def prepare_trace(x: np.ndarray, bleach_window: int=301) -> np.ndarray:
    y = correct_photobleaching(x, window=bleach_window)
    return robust_zscore(y)

def prepare_dataset_matrix(X: np.ndarray, bleach_window: int=301) -> np.ndarray:
    X = np.asarray(X, float)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        out[:, j] = prepare_trace(X[:, j], bleach_window=bleach_window)
    return out

def prepare_trace_with_baseline(x: np.ndarray, bleach_window: int=301, percentile: float=15.0):
    """Return (prepared_trace, estimated_baseline, corrected_trace_before_zscore)."""
    x = np.asarray(x, float)
    x_interp = interpolate_nans(x)
    b = estimate_bleach_baseline(x_interp, window=bleach_window, percentile=percentile)
    corrected = x_interp - b
    z = robust_zscore(corrected)
    return z, b, corrected
