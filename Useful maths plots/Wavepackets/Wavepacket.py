##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import scipy as sp
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
π = np.pi
ωs = np.linspace(1, 1.2, 10)
ts = np.linspace(-100 * π, 100 * π, 1000)

sins = []
for i0, val0 in enumerate(ωs):
    sins.append(np.cos(val0 * ts))

print(sum(sins))
##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

###

fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
for i0, val0 in enumerate(ωs):
    plt.plot(ts, sins[i0] + 2 * i0, '-')
plt.tight_layout()

###

fig2 = plt.figure('fig2', figsize=(5, 5))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(sum(sins), '.:')
plt.tight_layout()

###

plt.show()
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
