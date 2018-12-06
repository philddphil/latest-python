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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)"
      r"\SCM Data 20181113"
      r"\PSat 181839")

# Load data
Ps, cps = prd.load_Psat(p0)

# Scale cps
kcps = cps / 1000

# Perform fit (I_sat, P_sat, P_bkg, bkg)
initial_guess = (1.5e2, 1e-1, 1e3, 1e0)
popt, _ = opt.curve_fit(prd.I_sat, Ps, kcps, p0=initial_guess,
                        bounds=((0, 0, 0, 0),
                                (np.inf, np.inf, np.inf, np.inf)))
Ps_fit = np.linspace(np.min(Ps), np.max(Ps), 1000)
Isat_fit = prd.I_sat(Ps_fit, *popt)

I_sat = np.round(popt[0])
P_sat = np.round(popt[1] * 1000)
Prop_bkg = np.round(popt[2])
bkg = np.round(popt[3])

lb0 = 'fit'
lb1 = 'emitter I$_{sat}$ = ' + str(I_sat) + 'kcps'
lb2 = 'bkg, + ' + str(bkg) + 'kcps offset'
lb3 = 'I$_{sat}$ = ' + str(I_sat) + 'kcps, ' + \
    'P$_{sat}$ = ' + str(P_sat) + 'μW'

##############################################################################
# Plot some figures
##############################################################################

fig2 = plt.figure('fig2', figsize=(6, 4))
prd.ggplot()
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Inferred power (mW)')
ax2.set_ylabel('kcounts per secound')
plt.plot(Ps, kcps, 'o:', label='data')
plt.plot(Ps_fit, Isat_fit, '-', label=lb0)
plt.plot(Ps_fit, prd.I_sat(Ps_fit, popt[0], popt[1], 0, popt[3]),
         '--', label=lb1)
plt.plot(Ps_fit, prd.I_sat(Ps_fit, 0, popt[1], popt[2], popt[3]),
         '--', label=lb2)

ax2.legend(loc='upper left', fancybox=True, framealpha=1)
os.chdir(p0)
plt.title(lb3)
plt.tight_layout()
plt.show()
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig2, ax2, 'Sat curve.png')
