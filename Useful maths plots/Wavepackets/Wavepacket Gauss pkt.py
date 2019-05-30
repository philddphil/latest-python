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
ts = np.linspace(-15 * π, 15 * π, 1000)
Es = prd.Gaussian_1D(ts, 1, 0, 8 * π, 0, 5) * np.cos(ts*2)

##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
plot_path = r"D:\Python\Plots\\"
plot_label = 'Gaussian wavepacket'
plot_file_name = plot_path + plot_label
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
plt.plot(ts, Es, '-')
plt.tight_layout()
plt.axis('off')

###

plt.savefig(plot_file_name + '.svg')
plt.show()
prd.PPT_save_2d(fig1, ax1, plot_file_name + '.png')
