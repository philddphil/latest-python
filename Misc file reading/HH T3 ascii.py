##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
# NPL laptop library path
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
# Surface Pro 4 library path
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Import data (saved by labVIEW code controlling HH400)
##############################################################################
d0 = r"C:\local files\Experimental Data\F5 L10 Confocal measurements" + \
    r"\SCM Data 20200117\HH\HH T3 164743"
p0 = d0 + r"\tt ch0.txt"
p1 = d0 + r"\tt ch1.txt"

tt0 = np.loadtxt(p0)
tt1 = np.loadtxt(p1)
# 1e-7 is the saved resolution - this is 0.1 microsecond
total_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-7
c0 = len(tt0)
c1 = len(tt1)
cps0 = c0 / total_t
cps1 = c1 / total_t
print('total time collected ', np.round(total_t, 5), 's')
print('Ctr 1 - ', np.round(cps0 / 1000, 2), ' k counts per second')
print('Ctr 2 - ', np.round(cps1 / 1000, 2), ' k counts per second')


##############################################################################
# Generate time series based on photon arrival times
##############################################################################
ts0 = np.zeros(int(total_t * 1e7))
ts1 = np.zeros(int(total_t * 1e7))

for i0, v0 in enumerate(tt0):
    ts0[int(v0) - 2] = 1

for i0, v0 in enumerate(tt1):
    ts1[int(v0) - 2] = 1


# ts0_d = sp.signal.decimate(ts0, 1000)
# ts1_d = sp.signal.decimate(ts1, 1000)

# print(np.shape(ts0_d), np.shape(ts1_d))
##############################################################################
# Generate time series based on photon arrival times
##############################################################################

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"

###### xy plot ###############################################################
# size = 4
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Δt (ps)')
# ax2.set_ylabel('freq #')
# plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
# plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)
# ax2.set_yscale('log')

# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Δt (ps)')
# ax2.set_ylabel('freq #')
# plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
# plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)

###### xyz plot ##############################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'], label='')
# surffit = ax3.contour(*coords, z, 10, cmap=cm.jet)

# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

# plt.show()
# ax2.figure.savefig('SPAD histogram' + '.jpg')
# plot_file_name = plot_path + 'plot1.png'
# prd_plots.PPT_save_3d(fig2, ax2, plot_file_name)
