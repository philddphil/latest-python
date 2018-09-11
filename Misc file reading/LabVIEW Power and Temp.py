##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
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
p0 = r"D:\LabVIEW Data\21082018 104028 - Before and after laser locking"
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
T = []
t = []
P1 = []
P2 = []
for i1, val in enumerate(datafiles[0:]):
    print('reading: ', i1, val)
    data = np.loadtxt(val)
    T = np.append(T, data[:, 3])
    t = np.append(t, data[:, 0])
    P1 = np.append(P1, data[:, 1])
    P2 = np.append(P2, data[:, 2])

# Crop data
print('cropping data')
T = T[0:]
t = t[0:]
P1 = P1[0:]
P2 = P2[0:]

# Remove void T values
for i1, val in enumerate(T):
    if val > 25 or val < 19:
        T[i1] = 22
        print(i1)

# Scale data
print('scaling data')
t_ms = t - np.min(t)
t_s = t_ms / 1000
t_hrs = t_s / 3600
ΔT = T - np.min(T)

print(T)
##############################################################################
# Plot some figures
##############################################################################

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

fig2 = plt.figure('fig2', figsize=(10, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('time, (hrs)')
ax2.set_ylabel('Power (μW)')
ax2.plot(t_hrs, P1, label='P1')
ax2.plot(t_hrs, P2, label='P2')
ax2.plot(t_hrs, P1 + P2, label='P$_{tot}$')
ax2.legend(loc='upper left', fancybox=True, framealpha=1)
ax2a = ax2.twinx()
ax2a.set_ylabel('Δ Temp. ($^\circ$C)', color=cs['mnk_orange'])
ax2a.plot(t_hrs, ΔT, '--', c=cs['mnk_orange'], lw=0.5, label='Temp.')
ax2a.tick_params(axis='y', labelcolor=cs['mnk_orange'])
ax2a.legend(loc='upper right', fancybox=True, framealpha=1)

os.chdir(p0)
plt.tight_layout()
plt.show()
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
ax2a.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig2, ax2, 'plot2a.png')
