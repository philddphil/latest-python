# %% 0. imports
import numpy as np
from scipy import constants
# 1. defs
def mode_freq(L,q,m,n,R):
    """Calculate higher-order mode frequencies

    Args:
        L (float): Cavity length
        q (int): longitudinal mode order
        m (int): 1st tranverse mode order
        n (int): 2nd transverse mode order
        R (float): Cavity RoC

    Returns:
        ν (float) : frequency of mode
    """
    g = 1 - L/R
    nu = ((constants.c / (2*L)) * 
         (q + (1/np.pi) * (m + n + 1) * 
          np.arccos(np.sqrt(g ** 2))))
    return nu 

# 2. stuff
nu_525_0_0 = mode_freq(360e-6, 525, 0, 0, 200e-6)
λ_525_0_0_nm = (constants.c/nu_525_0_0)*1e9
nu_525_1_0 = mode_freq(360e-6, 525, 1, 0, 200e-6)
λ_525_1_0_nm = (constants.c/nu_525_1_0)*1e9
nu_526_0_0 = mode_freq(360e-6, 526, 0, 0, 220e-6)
λ_526_0_0_nm = (constants.c/nu_526_0_0)*1e9

FSR_GHz = np.abs(nu_525_0_0 - nu_526_0_0)*1e-9

HOS_GHz = np.abs(nu_525_0_0 - nu_525_1_0)*1e-9
