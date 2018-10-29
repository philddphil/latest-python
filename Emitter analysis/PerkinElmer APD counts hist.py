##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20181009"
      r"\APD counts 161440 Laser min, ND5 in")
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
t = []
Cts1 = []
Cts2 = []

for i0, val0 in enumerate(datafiles[:]):
    print('reading: ', i0, val0)
    lb = os.path.basename(val0)
    data = np.loadtxt(val0)
    t = np.append(t, data[:, 0])
    Cts1 = np.append(Cts1, data[:, 1])
    Cts2 = np.append(Cts2, data[:, 2])

# Crop data
print('cropping data')
# t = t[0:100]
# P1 = P1[0:100]
# P2 = P2[0:100]

# Scale data
print('scaling data')
t_ms = t - np.min(t)
t_s = t_ms / 1000
# t_hrs = t_s / 3600

# Stats on data
Cts1_μ = np.mean(Cts1)
Cts1_σ = np.sqrt(np.var(Cts1))
Cts2_μ = np.mean(Cts2)
Cts2_σ = np.sqrt(np.var(Cts2))

##############################################################################
# Plot some figures
##############################################################################

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Count rate')
ax2.set_ylabel('Freq (#)')
lb1 = 'kcps 1 = ' + str(np.round(Cts1_μ/1e3,1))
lb2 = 'kcps 2 = ' + str(np.round(Cts2_μ/1e3,1))
lb3 = 'kcps$_{tot}$ = ' + str(np.round((Cts1_μ + Cts2_μ)/1e3,1))
Cts1_n, Cts1_bins, Cts1_patches = plt.hist(Cts1, bins=10, label=lb1,
                                           alpha=0.5, edgecolor=cs['mnk_dgrey'])

Cts2_n, Cts2_bins, Cts2_patches = plt.hist(Cts2, bins=10, label=lb2,
                                           alpha=0.5, edgecolor=cs['mnk_dgrey'])

Cts3_n, Cts3_bins, Cts3_patches = plt.hist(Cts2 + Cts1, bins=10, label=lb3,
                                           alpha=0.5, edgecolor=cs['mnk_dgrey'])

x1, y1 = prd.Gauss_hist(Cts1)
x2, y2 = prd.Gauss_hist(Cts2)
x3, y3 = prd.Gauss_hist(Cts1 + Cts2)

plt.plot(x1, y1, '-', linewidth=2, color=cs['ggred'])
plt.plot(x2, y2, '-', linewidth=2, color=cs['ggblue'])
plt.plot(x3, y3, '-', linewidth=2, color=cs['ggpurple'])

ax2.legend(loc='upper left', fancybox=True, framealpha=0.2)

os.chdir(p0)
plt.tight_layout()
plt.show()
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.3))

prd.PPT_save_2d(fig2, ax2, 'hists.png')
