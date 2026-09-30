"""Stage-one full-band OCT reconstruction pipeline."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .calibration import make_calibration
from .dispersion import manual_dispersion_phase
from .fringe import balance_fringes
from .reconstruction import reconstruct_oct
from .stft import calculate_stft_windows
from matplotlib import pyplot as plt

from .config import (
    BIG_WINDOW_RESOLUTION_M,
    DISPERSION_C2,
    DISPERSION_C3,
    FIRST_WAVELENGTH_M,
    LAST_WAVELENGTH_M,
)


def read_tiff_spectrum(path: str | Path) -> np.ndarray:
    """Read a raw TIFF in its stored ``(A-lines, detector_pixels)`` orientation."""
    with Image.open(path) as image:
        data = np.asarray(image)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D raw TIFF, got {data.shape} from {path}")
    return data


def load_pixel_map(mat_path: str | Path) -> np.ndarray:
    """Load the ``pixelMap`` variable from the provided calibration MAT file."""
    contents = loadmat(mat_path)
    if "pixelMap" not in contents:
        raise KeyError(f"pixelMap not found in {mat_path}")
    return np.asarray(contents["pixelMap"]).squeeze()


def reconstruct_full_band_pair(
    s1_path: str | Path,
    s2_path: str | Path,
    pixel_map_path: str | Path,
    *,
    c2: float = DISPERSION_C2,
    c3: float = DISPERSION_C3,
) -> tuple[np.ndarray, dict]:
    """Balance one S1/S2 TIFF pair and reconstruct its full-band OCT B-scan."""
    s1 = read_tiff_spectrum(s1_path)
    s2 = read_tiff_spectrum(s2_path)
    if s1.shape != s2.shape:
        raise ValueError(f"S1 and S2 shapes differ: {s1.shape} versus {s2.shape}")

    # MATLAB stacks TIFFs vertically and then transposes: 400 x 2048 becomes
    # detector-pixels x A-lines = 2048 x 400 for a single TIFF pair.
    raw1 = s1.astype(np.float32).T
    raw2 = s2.astype(np.float32).T
    pixel_map = load_pixel_map(pixel_map_path)
    raw_balanced = balance_fringes(raw1, raw2, pixel_map)
    calibration = make_calibration(raw_balanced.shape[0])
    phase = manual_dispersion_phase(calibration.wavenumber_linear, c2, c3)
    full_band = calculate_stft_windows(
        calibration.wavenumber_linear,
        res_axis=raw_balanced.shape[0],
        up_sample_factor=1,
        win_num=1,
        resol_m=BIG_WINDOW_RESOLUTION_M,
        first_wave_m=FIRST_WAVELENGTH_M,
        last_wave_m=LAST_WAVELENGTH_M,
    )
    oct_bscan = reconstruct_oct(
        raw_balanced,
        calibration.wavenumber,
        calibration.wavenumber_linear,
        full_band.windows[:, :1],
        phase,
    )
    metadata = {
        "raw_tiff_shape": list(s1.shape),
        "raw_spectrum_shape": list(raw_balanced.shape),
        "oct_bscan_shape": list(oct_bscan.shape),
        "dispersion_c2": c2,
        "dispersion_c3": c3,
        "pixel_map_min": float(pixel_map.min()),
        "pixel_map_max": float(pixel_map.max()),
    }
    return oct_bscan, metadata
