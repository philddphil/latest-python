##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
# NPL laptop library path
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
# Surface Pro 4 library path
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
# PXI path
sys.path.insert(0, r"C:\Users\ormphotons\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Import data (saved by labVIEW code controlling HH400)
##############################################################################
d0 = r"C:\Data\SCM\20210819 SCM Data\HH T3 141102"
p0 = d0 + r"\tt ch0.txt"
p1 = d0 + r"\tt ch1.txt"

tt0 = np.loadtxt(p0)
tt1 = np.loadtxt(p1)

total_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-7
c0 = len(tt0)
c1 = len(tt1)
cps0 = c0 / total_t
cps1 = c1 / total_t
print(total_t)
print(c0)
print(c1)
print(np.round(cps0))
print(np.round(cps1))
print

##############################################################################
# Generate time series based on photon arrival times
##############################################################################
# ts1 = np.zeros(int(np.max(ts_1c)))
# ts2 = np.zeros(int(np.max(ts_2c)))
# # print(np.shape(ts1))
# # print(np.shape(ts2))
# for i0, v0 in enumerate(ts_1c):
#     ts1[int(v0) - 1] = 1

# for i0, v0 in enumerate(ts_2c):
#     ts2[int(v0) - 1] = 1

# print(np.shape(ts1))

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
