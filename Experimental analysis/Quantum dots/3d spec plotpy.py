##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# Specify results directory and change working directory to this location
p0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Conferences\SPW 2019\Data\Spectra\QD powers")
# p0 = (r"D:\Experimental Data\Internet Thorlabs optics data"))
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
roi = 80

λs, ctss, lbs = prd_file_import.load_spec_dir(p0)
xs0 = λs[1]
ys0 = ctss[1]

datafiles = glob.glob(p0 + r'\Power series.dat')
datafile = datafiles[0]
Ps = np.genfromtxt(datafile, skip_header=1)
Pwrs = []
for i0, v0 in enumerate(lbs):
    λs[i0] = λs[i0][200:]
    ctss[i0] = ctss[i0][200:]
    Pwrs.append(Ps[i0]*np.ones(len(λs[0])))

prd_plots.ggplot()
size = 4
fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
ax3 = fig3.add_subplot(111, projection='3d')
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('wavelength, λ (nm)')
ax3.set_ylabel('excitation power (μW)')
ax3.set_zlabel('Spectral counts (arb.)')
for i0, v0 in enumerate(lbs):
    scatexp = ax3.plot(λs[i0], Pwrs[i0], ctss[i0], '-', alpha=0.8, color=cs['gglred'], label='')

# os.chdir(p0)
plt.tight_layout()
ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.set_xlim3d(925, 945)
# ax.set_ylim3d(0,1000)
# ax.set_zlim3d(0,1000
plt.box()
plt.show()
prd_plots.PPT_save_3d(fig3, ax3, 'test.svg')
# set_zlim(min_value, max_value)
