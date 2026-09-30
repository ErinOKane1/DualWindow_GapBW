"""Spectrometer fringe matching and balancing."""

import numpy as np
from scipy.interpolate import CubicSpline


def balance_fringes(raw1: np.ndarray, raw2: np.ndarray, pixel_map: np.ndarray) -> np.ndarray:
    """Port of ``balanceFringes.m``.

    Parameters are detector-by-A-line arrays. ``pixel_map`` contains MATLAB
    one-based detector coordinates for S1, mapped onto the S2 detector axis.
    """
    raw1 = np.asarray(raw1, dtype=np.float32)
    raw2 = np.asarray(raw2, dtype=np.float32)
    if raw1.shape != raw2.shape or raw1.ndim != 2:
        raise ValueError("raw1 and raw2 must be two-dimensional arrays of equal shape")

    res_axis = raw1.shape[0]
    pixel_map = np.asarray(pixel_map, dtype=np.float64).squeeze()
    if pixel_map.shape != (res_axis,):
        raise ValueError(f"pixel_map must have shape ({res_axis},), got {pixel_map.shape}")

    # Evaluate only in the valid MATLAB interpolation interval. MATLAB zeros
    # points where pixelMap <= 0 or >= res_axis after interpolation.
    valid = (pixel_map > 0) & (pixel_map < res_axis)
    raw1_interpolated = np.zeros_like(raw1, dtype=np.float32)
    interpolator = CubicSpline(np.arange(1, res_axis + 1), raw1, axis=0)
    raw1_interpolated[valid] = interpolator(pixel_map[valid]).astype(np.float32)

    background1 = raw1_interpolated.mean(axis=1, keepdims=True)
    background2 = raw2.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = background2 / background1
        balanced = raw1_interpolated * ratio - raw2
    balanced[~valid] = 0
    return np.nan_to_num(balanced, copy=False).astype(np.float32, copy=False)
