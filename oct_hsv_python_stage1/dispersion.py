"""Manual dispersion phase compensation."""

import numpy as np


def manual_dispersion_phase(wavenumber_linear: np.ndarray, c2: float, c3: float) -> np.ndarray:
    """Exact active calculation in ``calculate_dispersion_compensation_phase_manual.m``."""
    n = np.linspace(-1, 1, np.asarray(wavenumber_linear).size)
    return (c2 * n**2 + c3 * n**3).astype(np.float32)
