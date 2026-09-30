"""Full-band and Gaussian spectral-window generation."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class STFTWindows:
    windows: np.ndarray
    wavelength_centers_m: np.ndarray


def calculate_stft_windows(
    wavenumber_linear: np.ndarray,
    res_axis: int,
    up_sample_factor: int,
    win_num: int,
    resol_m: float,
    first_wave_m: float,
    last_wave_m: float,
) -> STFTWindows:
    """Port of ``calculateSTFTWindows_k_v2.m``.

    Column zero is the Tukey full-band window; remaining columns are Gaussian
    STFT windows.
    """
    x = np.asarray(wavenumber_linear, dtype=np.float64)
    if x.ndim != 1 or x.size != res_axis * up_sample_factor:
        raise ValueError("wavenumber_linear length must equal res_axis * up_sample_factor")

    tukey_window = tukey(res_axis, alpha=0.25)
    source = np.linspace(1, res_axis, res_axis)
    query = np.linspace(1, res_axis, res_axis * up_sample_factor)
    full_band = interp1d(source, tukey_window, kind="linear")(query)

    last_k = 2 * np.pi / first_wave_m
    first_k = 2 * np.pi / last_wave_m
    k_centers = np.linspace(first_k, last_k, win_num)
    wavelength_centers_m = 2 * np.pi / k_centers
    sigma = 1 / (np.sqrt(2) * resol_m)

    windows = np.empty((x.size, win_num + 1), dtype=np.float32)
    windows[:, 0] = full_band
    # Evaluate all Gaussian windows at once; this removes Python-loop overhead
    # during one-time reconstruction setup without changing their definition.
    gaussians = np.exp(-0.5 * ((x[:, None] - k_centers[None, :]) / sigma) ** 2)
    gaussians -= gaussians.min(axis=0, keepdims=True)
    gaussians /= gaussians.max(axis=0, keepdims=True)
    windows[:, 1:] = gaussians.astype(np.float32, copy=False)

    return STFTWindows(windows=windows, wavelength_centers_m=wavelength_centers_m)
