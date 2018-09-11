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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)"
      r"\Noise eater implementation\Data 10092018"
      r"\Tholabs PM100")
name0 = r"\Hi power 2.txt"
name1 = r"\Hi power 1.txt"

f0 = p0 + name0
f1 = p0 + name1

d0 = open(f0, 'r', encoding='utf-8')
x0 = d0.readlines()
d0.close()
dts0 = []
Ps0 = []
for i0, val in enumerate(x0[2:]):
    t = val.split("\t")[0]
    ts0 = datetime.strptime(t, "%d/%m/%Y %H:%M:%S.%f   ")
    dts0 = np.append(dts0, ts0.timestamp() * 1000)
    Ps0 = np.append(Ps0, float(val.split("\t")[1]))
dts0 = dts0 - np.min(dts0)
Ps0_bar = Ps0 - np.mean(Ps0)
P0_avg = np.round(1e3 * np.mean(Ps0), 3)
l0 = ('Avg P = ' + str(P0_avg) + 'mW')

d1 = open(f1, 'r', encoding='utf-8')
x1 = d1.readlines()
d1.close()
dts1 = []
Ps1 = []
for i0, val in enumerate(x1[2:]):
    t = val.split("\t")[0]
    ts1 = datetime.strptime(t, "%d/%m/%Y %H:%M:%S.%f   ")
    dts1 = np.append(dts1, ts1.timestamp() * 1000)
    Ps1 = np.append(Ps1, float(val.split("\t")[1]))
dts1 = dts1 - np.min(dts1) + np.max(dts0)
Ps1_bar = Ps1 - np.mean(Ps1)
P1_avg = np.round(1e3 * np.mean(Ps1), 3)
l1 = ('Avg P = ' + str(P1_avg) + 'mW')

d2 = open(f2, 'r', encoding='utf-8')
x2 = d2.readlines()
d2.close()
dts2 = []
Ps2 = []
for i0, val in enumerate(x1[2:]):
    t = val.split("\t")[0]
    ts1 = datetime.strptime(t, "%d/%m/%Y %H:%M:%S.%f   ")
    dts1 = np.append(dts1, ts1.timestamp() * 1000)
    Ps1 = np.append(Ps1, float(val.split("\t")[1]))
dts1 = dts1 - np.min(dts1) + np.max(dts0)
Ps1_bar = Ps1 - np.mean(Ps1)
P1_avg = np.round(1e3 * np.mean(Ps1), 3)
l1 = ('Avg P = ' + str(P1_avg) + 'mW')
##############################################################################
# Plot some figures
##############################################################################

fig2 = plt.figure('fig2', figsize=(10, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('time, (s)')
ax2.set_ylabel('Power (mW)')
ax2.plot(dts0, 1e3 * Ps0_bar, '.', label=l0)
ax2.plot(dts1, 1e3 * Ps1_bar, '.', label=l1)
ax2.legend(loc='upper left', fancybox=True, framealpha=1)

plt.tight_layout()
plt.show()
