##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ntpath
from datetime import datetime

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
p0 = (r"D:\Experimental Data\Spectrometer (Ganni F5 L10)\Data 20180918")
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)

fig1 = plt.figure('fig1', figsize=(10, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Counts')

for i0, val0 in enumerate(datafiles[2:]):
    λ, cts = prd.load_Horiba(val0)
    lb = os.path.basename(val0)
    ax1.plot(λ, cts, '.', label=lb)

ax1.legend(loc='lower right', fancybox=True, framealpha=1)
plt.tight_layout()
plt.show()

ax1.legend(loc='lower right', fancybox=True, framealpha=0)
os.chdir(p0)
prd.PPT_save_2d(fig1, ax1, 'PCF spectrum.png')
