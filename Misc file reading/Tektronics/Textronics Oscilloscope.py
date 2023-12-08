##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\Oscilloscope (F5 L10)\SPAD outputs"
      r"\SPAD2.csv")
lb = os.path.basename(p0)
name = os.path.splitext(p0)[0]
csv = np.genfromtxt(p0, delimiter=",")
t = 1e6 * csv[:, 3]
V = csv[:, 4]


# plot each data set and save (close pop-up to save each time)
prd.ggplot()
fig1 = plt.figure('fig1', figsize=(6, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('time / μs')
ax1.set_ylabel('Voltage / V')
ax1.plot(t, V, lw=0.5)
ax1.plot(t, V, '.', markersize=0.3, alpha=0.5, label=lb)

plt.title('time trace')
plt.tight_layout()
ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
plt.show()
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig1, ax1, name)
