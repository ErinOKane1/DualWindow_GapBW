"""CPU reference port of the active OCT reconstruction GPU function."""

import numpy as np
from scipy.interpolate import CubicSpline


def reconstruct_oct(
    raw: np.ndarray,
    wavenumber: np.ndarray,
    wavenumber_linear: np.ndarray,
    stft_windows: np.ndarray,
    dispersion_phase: np.ndarray,
    *,
    up_sample_factor: int = 1,
    alines_per_chunk: int = 1024,
) -> np.ndarray:
    """Port ``oct_reconstruction_gpu.m`` using NumPy/SciPy on CPU.

    Returns ``(depth, A-lines)`` for one window and
    ``(depth, A-lines, windows)`` otherwise. The return value is an amplitude
    image, as MATLAB applies ``abs`` at the end of the function.
    """
    raw = np.asarray(raw, dtype=np.float32)
    stft_windows = np.asarray(stft_windows, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] % 2:
        raise ValueError("raw must have an even detector-axis length and shape (pixels, A-lines)")
    res_axis, total_alines = raw.shape
    output_depth = res_axis // 2
    expected_points = res_axis * up_sample_factor
    if stft_windows.shape[0] != expected_points:
        raise ValueError("stft_windows has an incompatible detector-axis length")
    if np.asarray(dispersion_phase).shape != (expected_points,):
        raise ValueError("dispersion_phase has an incompatible detector-axis length")

    n_windows = stft_windows.shape[1]
    result = np.empty((output_depth, total_alines, n_windows), dtype=np.complex64)
    phase_term = np.exp(-1j * np.asarray(dispersion_phase, dtype=np.float32)).astype(np.complex64)

    for start in range(0, total_alines, alines_per_chunk):
        stop = min(start + alines_per_chunk, total_alines)
        fringe = raw[:, start:stop].copy()
        fringe[:2] = 0
        fringe[-3:] = 0

        fringe_fft = np.fft.fftshift(np.fft.fft(fringe, axis=0), axes=0)
        center = res_axis // 2
        # MATLAB indices res_axis/2-8 : res_axis/2+8 become these zero-based indices.
        fringe_fft[center - 9:center + 8] = 0
        fringe_fft = np.fft.fftshift(fringe_fft, axes=0)

        if up_sample_factor > 1:
            pad = (up_sample_factor - 1) * res_axis // 2
            fringe_fft = np.pad(fringe_fft, ((pad, pad), (0, 0)), mode="constant")
        fringe_k = np.real(np.fft.ifft(fringe_fft, axis=0))

        # MATLAB: interp1(wn', flipud(fringe_fft_gpu), wn_lin', 'spline')
        # wn is descending in the MATLAB script, so reverse it to present an
        # increasing x-axis to SciPy's spline implementation.
        y = np.flipud(fringe_k)
        k = np.asarray(wavenumber, dtype=np.float64)
        if np.any(np.diff(k) < 0):
            k, y = k[::-1], y[::-1]
        fringe_linear = CubicSpline(k, y, axis=0)(np.asarray(wavenumber_linear, dtype=np.float64))

        for band in range(n_windows):
            fringe_windowed = fringe_linear * stft_windows[:, band, None] * phase_term[:, None]
            result[:, start:stop, band] = np.fft.fft(fringe_windowed, axis=0)[:output_depth]

    amplitude = np.abs(result)
    return amplitude[:, :, 0] if n_windows == 1 else amplitude
