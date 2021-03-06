##############################################################################
# Import some libraries
##############################################################################
import sys
import glob
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
    r"\SCM Data 20200128\HH\HH T3 123912"
p0 = d0 + r"\0 tt ch0.txt"
p1 = d0 + r"\1 tt ch1.txt"

datafiles0 = glob.glob(d0 + r'\*0.txt')
datafiles1 = glob.glob(d0 + r'\*1.txt')
HBT_ss = []
prd_plots.ggplot()
size = 4

for i0, v0 in enumerate(datafiles0[0:1]):
    print('file #', i0, ' ', v0)
    # 1e-7 is the saved resolution - this is 0.1 microsecond
    tta = np.loadtxt(datafiles0[i0])
    ttb = np.loadtxt(datafiles1[i0])

    # conditional conversion of the time series.
    # longer one is the start channel
    if len(tta) > len(ttb):
        tt0_ns = [j0 * 1e2 for j0 in tta]
        tt1_ns = [j0 * 1e2 for j0 in ttb]
    # convert to ns
    else:
        tt0_ns = [j0 * 1e2 for j0 in ttb]
        tt1_ns = [j0 * 1e2 for j0 in tta]

    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns)]) * 1e-9
    cs0 = len(tt0_ns)
    cs1 = len(tt1_ns)

    cps0 = cs0 / total_t
    cps1 = cs1 / total_t

    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)

    idx1 = np.arange(cs1)

    print('total time collected ', np.round(total_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')

    ##########################################################################
    # Perform start-stop measurements
    ##########################################################################
    locale = 1500
    tt_pad = np.pad(tt1_ns, locale + 1)
    tt_x = np.arange(0, len(tt_pad)) - locale + 1

    # loop through tt0_ns as our start channel
    for i1, v1 in enumerate(tt0_ns[0:]):
        chk = i1 // 10000
        i1 = i1 + 10000
        v1 = tt0_ns[i1]
        # trunkate full tt1_ns list to a region of 200 values around the same
        # time as the value v0 is (need to use dydx1 to convert)

        # 1. find the corresponding index in tt1_ns
        i_tt1 = int(v1 / dydx1)

        # 2.  specify locale around idx to check - get both times & idx vals
        tt_local = tt_pad[i_tt1:i_tt1 + 2 * locale]
        x_local = np.arange(i_tt1, i_tt1 + 2 * locale) - locale + 1

        # substract ith value of tt0_ns (v1) from tt1_ns & use abs operator
        tt_temp = np.abs(tt_local - v1)

        # find idx of min
        HBT_idx0 = np.argmin(tt_temp)

        # find time difference of min value
        HBT_test0 = tt_local[HBT_idx0] - v1

        # if time difference within range, analyse sub-region
        if -1500 < HBT_test0 < 1500:
            print(HBT_idx0)
            # need loop to check another sub-region, around the minimum value
            for i2 in np.arange(-5, 5):
                # find idx of min
                HBT_idx1 = HBT_idx0 + i2

                # find time difference of min value
                HBT_test1 = tt_local[HBT_idx1] - v1

                # check if value is of interest
                if -1500 < HBT_test1 < 1500:
                    HBT_ss.append(HBT_test1)
                    if chk == 0 & i2 == 0:
                        print(i1)
                        fig1 = plt.figure('fig1', figsize=(
                            size * np.sqrt(2), size))
                        fig1.patch.set_facecolor(cs['mnk_dgrey'])
                        ax1 = fig1.add_subplot(111)

                        plt.plot(tt0_ns, '.',
                                 alpha=0.8,
                                 markersize=5,
                                 label='t0',
                                 mfc=cs['ggred'])
                        plt.plot([0, np.max(cs0)], [0, dydx0 * np.max(cs0)],
                                 lw=0.5,
                                 color=cs['gglred'])
                        plt.plot(tt_x, tt_pad, '.',
                                 alpha=0.8,
                                 markersize=5,
                                 label='t1',
                                 mfc=cs['ggblue'])
                        plt.plot([0, np.max(cs1)], [0, dydx1 * np.max(cs1)],
                                 lw=0.5,
                                 color=cs['gglblue'])
                        plt.plot(x_local, tt_temp, '.--',
                                 lw=0.5, alpha=1, markersize=5,
                                 label='|t1 - v0|')
                        plt.plot(i1, v1, 'o', mec=cs[
                                 'ggyellow'], mfc='none', label='v0')
                        plt.plot(x_local[HBT_idx1], tt_temp[HBT_idx1], 'o',
                                 mec=cs['mnk_orange'],
                                 mfc='none',
                                 label='HBT value')
                        plt.plot([i_tt1 - 30, i_tt1 + 30], [1500, 1500])
                        ax1.legend(loc='upper left',
                                   fancybox=True, framealpha=0.5)
                        # ax1.set_ylim(0, 1.5 * v0)
                        # ax1.set_xlim(i_tt1 - 30, i_tt1 + 30)
                        plt.show()

    print(np.shape(HBT_ss))
##############################################################################
# Plot some figures
##############################################################################
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
prd_plots.ggplot()
# ###### xy plot #########################################################
# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.hist(HBT_ss, 500)
# plt.title('')
# fig1.tight_layout()
# plt.show()

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
