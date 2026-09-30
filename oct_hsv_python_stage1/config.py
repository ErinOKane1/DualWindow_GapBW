"""Central processing parameters for the OCT/HSV pipeline."""

HUE_CORRELATION_MIN_UM = 4.0
HUE_CORRELATION_MAX_UM = 12.0
HUE_COLOR_SPAN = 0.75
VALUE_FLOOR_FRACTION = 0.66


# Dispersion compensation
DISPERSION_C3 = -54.63
DISPERSION_C2 = 38.02

# Spectral-window setup
FIRST_WAVELENGTH_M = 805e-9
LAST_WAVELENGTH_M = 983e-9
WINDOW_COUNT = 250

# MATLAB-matching STFT resolutions
SMALL_WINDOW_RESOLUTION_M = 300e-6
BIG_WINDOW_RESOLUTION_M = 10e-6

# HSV / structural threshold
INTENSITY_THRESHOLD_DB = 40.0

# Reconstruction
UPSAMPLE_FACTOR = 1
ALINES_PER_CHUNK = 1024