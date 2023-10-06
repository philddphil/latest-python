# %% imports
"""
Created on Wed Jun  8 11:59:46 2022

@author: pd10
"""
import numpy as np
import scipy as sp
from scipy import constants
# %% defs
def dfreq_to_dlambda(wavelength_nm, dfreq_GHz, rnd=3):
    """

    Parameters
    ----------
    wavelength : float, nm
        wavelength region.
    dfreq : float, GHz
        freqency range.
    rnd : interger
        number of decimal points to round the answer to

    Returns
    -------
    dlambda : float, nm
        wavelength range.

    """
    c = 299792458
    e = 1.60217662e-19
    h = 6.62607015e-34

    wavelength_m = wavelength_nm * 1e-9
    dfreq_Hz = dfreq_GHz * 1e9
    dnrg_J = dfreq_Hz * h
    dnrg_eV = dnrg_J / e

    freq_Hz = c / wavelength_m
    freq_GHz = freq_Hz/1e9

    dlambda_m = (c * dfreq_Hz) / (freq_Hz ** 2)
    dlambda_nm = dlambda_m * 1e9
    return np.round(dlambda_nm, rnd), np.round(dnrg_eV, rnd)

c = constants.c
e = constants.e 
h = constants.h
pi = np.pi

photon_rate_cps = 3e6
wavelength_nm = 1030
dfreq_GHz = 10

n1 = 1.45
d1 = (c / n1) * 12.5e-9

photon_E_j = (h*c)/(wavelength_nm*1e-9)
wavelength_m = wavelength_nm * 1e-9
dfreq_Hz = dfreq_GHz * 1e9
dnrg_J = dfreq_Hz * h
dnrg_eV = dnrg_J / e
dnrg_meV = dnrg_eV * 1e3
dnrg_μeV = dnrg_meV * 1e3

freq_Hz = c / wavelength_m
freq_GHz = freq_Hz / 1e9

dlambda_m = (c * dfreq_Hz) / (freq_Hz ** 2)
dlambda_nm = dlambda_m * 1e9
dlambda_pm = dlambda_nm * 1e3

# Coherence time, length & delay line
coh_len_m = c/(np.pi*dfreq_Hz)
coh_time_s = coh_len_m/c
coh_time_ns = 1e9*(coh_len_m/c)
life_time_s = 1/(2*pi*dfreq_Hz)
life_time_ns = 1e9/(2*pi*dfreq_Hz)

# Photon energy metrics
photon_J = h * freq_Hz
photon_flux_W = photon_J * photon_rate_cps
photon_flux_pW = 1e12 * photon_flux_W

# Cavity metrics
l = 7.5e-3
FSR = 10e9
Q1 = wavelength_nm / dlambda_nm
Q2 = freq_Hz/dfreq_Hz
Q = 1e7
F = FSR / dfreq_Hz
Ringdown = Q/(freq_Hz * 2 * pi)
