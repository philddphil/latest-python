# %% imports
import numpy as np
import scipy as sp
from scipy import constants
# %% defs
print(constants.c)
lambda_nm = 1030
lambda_m = lambda_nm * 1e-9
nu_hz = constants.lambda2nu(lambda_m)
nu_Mhz = nu_hz * 1e-6

# spectral widths


# pulse parameters
pulse_t_ns = 100
pulse_t_s = pulse_t_ns * 1e-9

# single photon values
E_j = constants.h * nu_hz
PulseP_W = E_j/pulse_t_s
PulseP_pW = PulseP_W * 1e12


