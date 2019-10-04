##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import scipy.optimize as opt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20190912\PSats\12Sep19-010.txt")
p1 = (r"D:\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20190912\PSats")

data = np.genfromtxt(p0, delimiter=',', skip_header=1)
with open(p0) as f:
    content = f.readlines()
peakxy = content[0].rstrip()
Ps = data[0]
kcps = data[1] / 1000

initial_guess = (1.5e2, 1e-1, 1e3, 1e0)
popt, _ = opt.curve_fit(prd_maths.I_sat, Ps, kcps, p0=initial_guess,
                        bounds=((0, 0, 0, 0),
                                (np.inf, np.inf, np.inf, np.inf)))

Ps_fit = np.linspace(np.min(Ps), np.max(Ps), 1000)
Isat_fit = prd_maths.I_sat(Ps_fit, *popt)

I_sat = np.round(popt[0])
P_sat = np.round(popt[1] * 1000)
Prop_bkg = np.round(popt[2])
bkg = np.round(popt[3])

lb0 = 'fit'
lb1 = 'I$_{sat}$ = ' + str(I_sat) + 'kcps'
lb2 = 'bkg = ' + str(Prop_bkg) + 'P + ' + str(bkg)
lb3 = 'I$_{sat}$ = ' + str(I_sat) + 'kcps, ' + \
    'P$_{sat}$ = ' + str(P_sat) + 'μW @ ' + peakxy


##############################################################################
# Plot some figures
##############################################################################
fig2 = plt.figure('fig2', figsize=(6, 4))
prd_plots.ggplot()
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Inferred power (mW)')
ax2.set_ylabel('kcounts per secound')
plt.plot(Ps, kcps, 'o:', label='data')
plt.plot(Ps_fit, Isat_fit, '-', label=lb0)
plt.plot(Ps_fit, prd_maths.I_sat(Ps_fit, popt[0], popt[1], 0, popt[3]),
         '--', label=lb1)
plt.plot(Ps_fit, prd_maths.I_sat(Ps_fit, 0, popt[1], popt[2], popt[3]),
         '--', label=lb2)

ax2.legend(loc='lower right', fancybox=True, framealpha=1)
os.chdir(p1)
plt.title(lb3)
plt.tight_layout()
plt.show()
ax2.legend(loc='lower right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig2, ax2, peakxy + '.png')
