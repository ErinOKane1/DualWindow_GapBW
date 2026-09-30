"""Dual-window two-band OCT reconstruction.

This module reuses the normal OCT preprocessing path already used by the
project: fringe balancing, k-linearization, manual dispersion compensation,
and Fourier-domain reconstruction. It adds two dual spectral windows centered
near opposite ends of the calibrated spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hann

from .calibration import make_calibration
from .config import DISPERSION_C2, DISPERSION_C3
from .dispersion import manual_dispersion_phase
from .fringe import balance_fringes
from .pipeline import load_pixel_map, read_tiff_spectrum
from .reconstruction import reconstruct_oct


@dataclass(frozen=True)
class DualWindowSet:
    """Container for broad, narrow, and product dual windows."""

    fraction_axis: np.ndarray
    broad_windows: np.ndarray
    narrow_windows: np.ndarray
    dual_windows: np.ndarray
    wavelength_centers_m: np.ndarray
    labels: tuple[str, str]


@dataclass(frozen=True)
class DualWindowBscan:
    """Full-band and two-band reconstruction for one S1/S2 pair."""

    full_band: np.ndarray
    short_wavelength: np.ndarray
    long_wavelength: np.ndarray
    short_broad_window: np.ndarray
    short_narrow_window: np.ndarray
    long_broad_window: np.ndarray
    long_narrow_window: np.ndarray
    windows: DualWindowSet
    metadata: dict


def _finite_hann_window(
    fraction_axis: np.ndarray,
    *,
    center_fraction: float,
    full_width_fraction: float,
) -> np.ndarray:
    """Return a finite-support Hann window with width in spectrum fraction."""
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError("center_fraction must be between 0 and 1.")
    if not 0.0 < full_width_fraction <= 1.0:
        raise ValueError("full_width_fraction must be in (0, 1].")

    half_width = full_width_fraction / 2.0
    left = center_fraction - half_width
    right = center_fraction + half_width
    mask = (fraction_axis >= left) & (fraction_axis <= right)
    window = np.zeros_like(fraction_axis, dtype=np.float32)
    count = int(np.count_nonzero(mask))
    if count < 2:
        raise ValueError("Window width is too narrow for the detector sampling.")
    window[mask] = hann(count, sym=True).astype(np.float32)
    return window


def make_two_band_dual_windows(
    wavenumber_linear: np.ndarray,
    *,
    center_separation_fraction: float = 0.75,
    broad_width_fraction: float = 0.25,
    narrow_width_fraction: float = 0.05,
) -> DualWindowSet:
    """Create long- and short-wavelength dual-window products.

    The two window centers are separated by 75% of the total sampled spectrum by
    default, which places them at 12.5% and 87.5% of the linear-k axis. The
    broad and narrow finite-support Hann windows are multiplied and normalized
    per band before reconstruction.
    """
    k = np.asarray(wavenumber_linear, dtype=np.float64)
    if k.ndim != 1:
        raise ValueError("wavenumber_linear must be one-dimensional.")
    if not 0.0 < center_separation_fraction < 1.0:
        raise ValueError("center_separation_fraction must be in (0, 1).")

    fraction_axis = np.linspace(0.0, 1.0, k.size, dtype=np.float64)
    low_center = 0.5 - center_separation_fraction / 2.0
    high_center = 0.5 + center_separation_fraction / 2.0
    centers = np.array([low_center, high_center], dtype=np.float64)

    broad = np.column_stack(
        [
            _finite_hann_window(
                fraction_axis,
                center_fraction=float(center),
                full_width_fraction=broad_width_fraction,
            )
            for center in centers
        ]
    )
    narrow = np.column_stack(
        [
            _finite_hann_window(
                fraction_axis,
                center_fraction=float(center),
                full_width_fraction=narrow_width_fraction,
            )
            for center in centers
        ]
    )
    dual = broad * narrow
    dual /= np.maximum(dual.max(axis=0, keepdims=True), 1e-12)

    center_k = np.interp(centers, fraction_axis, k)
    wavelength_centers = 2.0 * np.pi / center_k
    order = np.argsort(wavelength_centers)
    short_index = int(order[0])
    long_index = int(order[-1])
    labels = ("short_wavelength", "long_wavelength")
    ordered = [short_index, long_index]

    return DualWindowSet(
        fraction_axis=fraction_axis.astype(np.float32),
        broad_windows=broad[:, ordered].astype(np.float32, copy=False),
        narrow_windows=narrow[:, ordered].astype(np.float32, copy=False),
        dual_windows=dual[:, ordered].astype(np.float32, copy=False),
        wavelength_centers_m=wavelength_centers[ordered].astype(np.float64, copy=False),
        labels=labels,
    )


def reconstruct_dual_window_pair(
    s1_path: str | Path,
    s2_path: str | Path,
    pixel_map_path: str | Path,
    *,
    c2: float = DISPERSION_C2,
    c3: float = DISPERSION_C3,
    broad_width_fraction: float = 0.25,
    narrow_width_fraction: float = 0.05,
    center_separation_fraction: float = 0.75,
) -> DualWindowBscan:
    """Reconstruct full-band, short-band, and long-band OCT from one TIFF pair."""
    s1 = read_tiff_spectrum(s1_path)
    s2 = read_tiff_spectrum(s2_path)
    if s1.shape != s2.shape:
        raise ValueError(f"S1 and S2 shapes differ: {s1.shape} versus {s2.shape}")

    raw_balanced = balance_fringes(
        s1.astype(np.float32).T,
        s2.astype(np.float32).T,
        load_pixel_map(pixel_map_path),
    )
    res_axis = raw_balanced.shape[0]
    calibration = make_calibration(res_axis)
    phase = manual_dispersion_phase(calibration.wavenumber_linear, c2, c3)
    windows = make_two_band_dual_windows(
        calibration.wavenumber_linear,
        center_separation_fraction=center_separation_fraction,
        broad_width_fraction=broad_width_fraction,
        narrow_width_fraction=narrow_width_fraction,
    )

    full_band_window = np.ones((res_axis, 1), dtype=np.float32)
    full_band = reconstruct_oct(
        raw_balanced,
        calibration.wavenumber,
        calibration.wavenumber_linear,
        full_band_window,
        phase,
    )
    dual_band = reconstruct_oct(
        raw_balanced,
        calibration.wavenumber,
        calibration.wavenumber_linear,
        windows.dual_windows,
        phase,
    )
    broad_band = reconstruct_oct(
        raw_balanced,
        calibration.wavenumber,
        calibration.wavenumber_linear,
        windows.broad_windows,
        phase,
    )
    narrow_band = reconstruct_oct(
        raw_balanced,
        calibration.wavenumber,
        calibration.wavenumber_linear,
        windows.narrow_windows,
        phase,
    )

    metadata = {
        "raw_tiff_shape": list(s1.shape),
        "raw_spectrum_shape": list(raw_balanced.shape),
        "full_band_shape": list(full_band.shape),
        "dual_band_shape": list(dual_band.shape),
        "dispersion_c2": float(c2),
        "dispersion_c3": float(c3),
        "broad_width_fraction": float(broad_width_fraction),
        "narrow_width_fraction": float(narrow_width_fraction),
        "center_separation_fraction": float(center_separation_fraction),
        "short_center_nm": float(windows.wavelength_centers_m[0] * 1e9),
        "long_center_nm": float(windows.wavelength_centers_m[1] * 1e9),
    }
    return DualWindowBscan(
        full_band=full_band,
        short_wavelength=dual_band[:, :, 0],
        long_wavelength=dual_band[:, :, 1],
        short_broad_window=broad_band[:, :, 0],
        short_narrow_window=narrow_band[:, :, 0],
        long_broad_window=broad_band[:, :, 1],
        long_narrow_window=narrow_band[:, :, 1],
        windows=windows,
        metadata=metadata,
    )


def reconstruct_dual_window_volume(
    s1_paths: Sequence[str | Path],
    s2_paths: Sequence[str | Path],
    pixel_map_path: str | Path,
    *,
    c2: float = DISPERSION_C2,
    c3: float = DISPERSION_C3,
    broad_width_fraction: float = 0.25,
    narrow_width_fraction: float = 0.05,
    center_separation_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DualWindowSet, list[dict]]:
    """Reconstruct full, short, and long OCT volumes from sorted TIFF pairs."""
    if not s1_paths:
        raise ValueError("At least one S1/S2 TIFF pair is required.")
    if len(s1_paths) != len(s2_paths):
        raise ValueError("S1 and S2 path list lengths must match.")

    full_slices = []
    short_slices = []
    long_slices = []
    metadata = []
    windows = None
    for s1_path, s2_path in zip(s1_paths, s2_paths):
        result = reconstruct_dual_window_pair(
            s1_path,
            s2_path,
            pixel_map_path,
            c2=c2,
            c3=c3,
            broad_width_fraction=broad_width_fraction,
            narrow_width_fraction=narrow_width_fraction,
            center_separation_fraction=center_separation_fraction,
        )
        full_slices.append(result.full_band)
        short_slices.append(result.short_wavelength)
        long_slices.append(result.long_wavelength)
        metadata.append(result.metadata)
        windows = result.windows

    return (
        np.stack(full_slices, axis=2),
        np.stack(short_slices, axis=2),
        np.stack(long_slices, axis=2),
        windows,
        metadata,
    )


def amplitude_to_db(image: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Convert OCT amplitude to dB with a small numerical floor."""
    return 20.0 * np.log10(np.maximum(np.asarray(image), floor))


def plot_dual_window_result(
    result: DualWindowBscan,
    *,
    output_path: str | Path | None = None,
    dynamic_range_db: float = 45.0,
) -> plt.Figure:
    """Plot windows, full-band OCT, dual-window OCT, and component windows."""
    images_db = {
        "full": amplitude_to_db(result.full_band),
        "short_dual": amplitude_to_db(result.short_wavelength),
        "long_dual": amplitude_to_db(result.long_wavelength),
        "short_broad": amplitude_to_db(result.short_broad_window),
        "short_narrow": amplitude_to_db(result.short_narrow_window),
        "long_broad": amplitude_to_db(result.long_broad_window),
        "long_narrow": amplitude_to_db(result.long_narrow_window),
    }
    vmax = float(np.percentile(images_db["full"], 99.8))
    vmin = vmax - dynamic_range_db

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    ax = axes[0, 0]
    x = result.windows.fraction_axis
    for index, label in enumerate(("short", "long")):
        center_nm = result.windows.wavelength_centers_m[index] * 1e9
        ax.plot(x, result.windows.broad_windows[:, index], "--", label=f"{label} broad ({center_nm:.0f} nm)")
        ax.plot(x, result.windows.narrow_windows[:, index], ":", label=f"{label} narrow ({center_nm:.0f} nm)")
        ax.plot(x, result.windows.dual_windows[:, index], "-", label=f"{label} dual")
    ax.set_title("Dual windows")
    ax.set_xlabel("Fraction of linear-k spectrum")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    panels = [
        (axes[0, 1], images_db["full"], "Full-band OCT"),
        (axes[0, 2], images_db["short_dual"], "Short wavelength: broad x narrow"),
        (axes[0, 3], images_db["long_dual"], "Long wavelength: broad x narrow"),
        (axes[1, 0], images_db["short_broad"], "Short wavelength: broad only"),
        (axes[1, 1], images_db["short_narrow"], "Short wavelength: narrow only"),
        (axes[1, 2], images_db["long_broad"], "Long wavelength: broad only"),
        (axes[1, 3], images_db["long_narrow"], "Long wavelength: narrow only"),
    ]
    for ax, image, title in panels:
        im = ax.imshow(image, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("A-line")
        ax.set_ylabel("Depth pixel")
        fig.colorbar(im, ax=ax, label="Amplitude (dB)")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
