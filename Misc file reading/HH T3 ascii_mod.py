##############################################################################
# Import some libraries
##############################################################################
import sys
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns)]) * 1e-9
    cs0 = len(tt0_ns)
    cs1 = len(tt1_ns)
    idx1 = np.arange(cs1)
    fit_tt1_ns = interp1d(tt1_ns, idx1)

    load_time = time.time() - loop_start_t

    cps0 = cs0 / total_t
    cps1 = cs1 / total_t
    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)
    print('total time collected ', np.round(total_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')

    ##########################################################################
    # Perform start-stop measurements
    ##########################################################################
    locale = 1000
    tt_pad = np.pad(tt1_ns, locale + 1)

    # loop through tt0_ns as our start channel
    for i1, v1 in enumerate(tt0_ns[0:-2]):
        # loc1 = np.interp(v1, tt1_ns, idx1)
        idx_tt1 = int(fit_tt1_ns(v1))

        for i2, v2 in enumerate(np.arange(-1, 1)):
            HBT_test1 = tt1_ns[idx_tt1 + i2] - v1
       # trunkate full tt1_ns list to a region of 200 values around the same

            if -500 < HBT_test1 < 500:
                HBT_ss.append(HBT_test1)

    proc_time = time.time() - loop_start_t
    print('HBTs (#)', len(HBT_ss))
    print('File load time (s) = ', np.round(load_time, 4))
    print('Processing time (s) = ', np.round(proc_time, 4))

##############################################################################
# Plot some figures
##############################################################################
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
prd_plots.ggplot()
###### xy plot ###############################################################
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.hist(HBT_ss, 500)
plt.title('')
fig1.tight_layout()
plt.show()

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
