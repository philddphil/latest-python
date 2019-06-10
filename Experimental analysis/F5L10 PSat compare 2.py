##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opt
from scipy.stats import norm

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20181108"
      r"\PSat 112358")
p1 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20181108"
      r"\PSat 112445")

# Load data
Ps_raw0, cps0 = prd.load_Psat(p0)
Ps_raw1, cps1 = prd.load_Psat(p1)

# Scale Ps and cps
Ps0 = 100 * Ps_raw0
kcps0 = cps0 / 1000

Ps1 = 100 * Ps_raw1
kcps1 = cps1 / 1000

# Perform fit
initial_guess = (1.5e2, 1e-1, 1e1, 1e0)
popt, _ = opt.curve_fit(prd.I_sat, Ps0, kcps0, p0=initial_guess,
                        bounds=((0, 0, 0, 0),
                                (np.inf, np.inf, np.inf, np.inf)))
Ps0_fit = np.linspace(np.min(Ps0), np.max(Ps0), 1000)
Isat0_fit = prd.I_sat(Ps0_fit, *popt)
I_sat0 = np.round(popt[0])
P_sat0 = np.round(popt[1] * 1000)
Prop_bkg0 = np.round(popt[2])
bkg0 = np.round(popt[3])

initial_guess = (1.5e2, 1e-1, 1e1, 1e0)
popt, _ = opt.curve_fit(prd.I_sat, Ps1, kcps1, p0=initial_guess,
                        bounds=((0, 0, 0, 0),
                                (np.inf, np.inf, np.inf, np.inf)))
Ps1_fit = np.linspace(np.min(Ps1), np.max(Ps1), 1000)
Isat1_fit = prd.I_sat(Ps1_fit, *popt)
I_sat1 = np.round(popt[0])
P_sat1 = np.round(popt[1] * 1000)
Prop_bkg1 = np.round(popt[2])
bkg1 = np.round(popt[3])

##############################################################################
# Plot some figures
##############################################################################

fig2 = plt.figure('fig2', figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Inferred power (mW)')
ax2.set_ylabel('k-counts per secound')
plt.plot(Ps0, kcps0, 'o--')
plt.plot(Ps1, kcps1, 'o--')
plt.plot(Ps0_fit, Isat0_fit, '-', color=cs['ggred'])
plt.plot(Ps1_fit, Isat1_fit, '-', color=cs['ggblue'])

ax2.legend(loc='upper left', fancybox=True, framealpha=1)
os.chdir(p0)
plt.title('comparison')
plt.tight_layout()
plt.show()
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig2, ax2, 'Sat curve comparison.png')
