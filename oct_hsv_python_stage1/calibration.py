"""Wavelength/wavenumber calibration used by the supplied MATLAB script."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Calibration:
    wavelength_m: np.ndarray
    wavenumber: np.ndarray
    wavenumber_linear: np.ndarray


def make_calibration(res_axis: int, up_sample_factor: int = 1) -> Calibration:
    """Replicate the active polynomial calibration in the MATLAB script."""
    if res_axis <= 0 or up_sample_factor < 1:
        raise ValueError("res_axis must be positive and up_sample_factor must be >= 1")

    c0, c1, c2, c3 = (
        777.6669232026112,
        0.10516422684727454,
        -1.774766913219079e-6,
        -8.509587624989255e-10,
    )
    detector_index_zero_based = np.linspace(0, res_axis - 1, res_axis)
    wavelength_m = (c3 * detector_index_zero_based**3 + c2 * detector_index_zero_based**2
                    + c1 * detector_index_zero_based + c0) * 1e-9

    # MATLAB: interp1(1:res_axis, wavelength, linspace(1, res_axis, ...), 'spline')
    query = np.linspace(1, res_axis, res_axis * up_sample_factor)
    wavelength_m = CubicSpline(np.arange(1, res_axis + 1), wavelength_m)(query)
    wavenumber = 2 * np.pi / wavelength_m
    wavenumber_linear = np.linspace(wavenumber[0], wavenumber[-1], len(wavenumber))

    return Calibration(
        wavelength_m=wavelength_m,
        wavenumber=np.flip(wavenumber),
        wavenumber_linear=np.flip(wavenumber_linear),
    )
