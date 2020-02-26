
##############################################################################
# Import some libraries
##############################################################################
import time
import os
import sys
import glob
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# Do some stuff
##############################################################################
d0 = r"C:\local files\Experimental Data\F5 L10 Confocal measurements" + \
    r"\SCM Data 20200128\HH\HH T3 123912"
p0 = d0 + r"\0 tt ch0.txt"
p1 = d0 + r"\1 tt ch1.txt"

datafiles0 = glob.glob(d0 + r'\*0.txt')
datafiles1 = glob.glob(d0 + r'\*1.txt')
HBT_ss = []

for i0, v0 in enumerate(datafiles0[0:1]):
    loop_start_t = time.time()
    # i0 = 2
    print('file #', i0, ' ', datafiles0[i0])
    # 1e-7 is the saved resolution - this is 0.1 microsecond
    tta = np.loadtxt(datafiles0[i0])
    ttb = np.loadtxt(datafiles1[i0])
    tt0_ns = [j0 * 1e2 for j0 in ttb]
    tt1_ns = [j0 * 1e2 for j0 in tta]
    load_time = time.time() - loop_start_t
    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns)]) * 1e-9
    cs0 = len(tt0_ns)
    cs1 = len(tt1_ns)
    idx1 = np.arange(cs1)
    interp_time = time.time()
    fit_tt1_ns = interp1d(tt1_ns, idx1)
    print('interp time = ', np.round(time.time() - interp_time, 4))
    cps0 = cs0 / total_t
    cps1 = cs1 / total_t
    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)
    print('total time collected ', np.round(total_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')
    size = 4
    prd_plots.ggplot()
    fig1 = plt.figure('fig1', figsize=(
        size * np.sqrt(2), size))
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1 = fig1.add_subplot(111)

    plt.plot(tt1_ns - np.arange(0, cs1) * dydx1, '.',
             alpha=0.8,
             markersize=5,
             label='resids0')
    ax1.legend(loc='upper left',
               fancybox=True, framealpha=0.5)
    # ax1.set_ylim(0, 1.5 * v0)
    # ax1.set_xlim(i_tt1 - 30, i_tt1 + 30)
    plt.show()
##############################################################################
# Plot some figures
##############################################################################
# prep colour scheme for plots and paths to save figs to
prd_plots.ggplot()
# NPL path
plot_path = r"C:\local files\Python\Plots"
# Surface Pro path
plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
# os.chdir(plot_path)

# xy plot ####################################################################
# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(ts, dBs, '.-')
# plt.title('res=' + str(hdr[0])
#           + 's  ' + 'dur=' + str(hdr[1])
#           + 's ' + '@' + str(hdr[2]))
# fig1.tight_layout()
# plt.show()

# hist/bar plot ##############################################################
# size = 9
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Country', fontsize=28, labelpad=80,)
# ax2.set_ylabel('Money (M$)', fontsize=28)
# plt.bar(1, 500, color=cs['ggred'])
# plt.bar(2, 1000, color=cs['ggblue'])
# plt.bar(3, 1275, color=cs['mnk_green'])
# plt.bar(4, 10000, color=cs['ggpurple'])
# ax2.set_xlim(0.5, 4.5)
# ax2.set_ylim(0, 11000)
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Δt (ps)')
# ax2.set_ylabel('freq #')
# plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
# plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)

# xyz plot ###################################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4,
#                       color=cs['gglred'], label='')
# surffit = ax3.contour(*coords, z, 10, cmap=cm.jet)
# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

# img plot ###################################################################
# size = 4
# fig4 = plt.figure('fig4', figsize=(size * np.sqrt(2), size))
# ax4 = fig4.add_subplot(1, 1, 1)
# fig4.patch.set_facecolor(cs['mnk_dgrey'])
# ax4.set_xlabel('x dimension (V)')
# ax4.set_ylabel('y dimension (V)')
# plt.title('')
# im4 = plt.imshow(img, cmap='magma')
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig4.colorbar(im4, cax=cax)

# save plot ###################################################################
# plt.show()
# ax2.figure.savefig('funding' + '.png')
# plot_file_name = plot_path + 'plot2.png'
# # prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
