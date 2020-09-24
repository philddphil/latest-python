##############################################################################
# Import some libraries
##############################################################################
import os
import re
import sys
import glob
import matplotlib
import numpy as np
import scipy as sp

import scipy.signal
import scipy.optimize as opt
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################################################
# Some defs
##############################################################################
# Custom palette for plotting ################################################
def palette():
    colours = {'mnk_purple': [145 / 255, 125 / 255, 240 / 255],
               'mnk_dgrey': [39 / 255, 40 / 255, 34 / 255],
               'mnk_lgrey': [96 / 255, 96 / 255, 84 / 255],
               'mnk_green': [95 / 255, 164 / 255, 44 / 255],
               'mnk_yellow': [229 / 255, 220 / 255, 90 / 255],
               'mnk_blue': [75 / 255, 179 / 255, 232 / 255],
               'mnk_orange': [224 / 255, 134 / 255, 31 / 255],
               'mnk_pink': [180 / 255, 38 / 255, 86 / 255],
               ####
               'rmp_dblue': [12 / 255, 35 / 255, 218 / 255],
               'rmp_lblue': [46 / 255, 38 / 255, 86 / 255],
               'rmp_pink': [210 / 255, 76 / 255, 197 / 255],
               'rmp_green': [90 / 255, 166 / 255, 60 / 255],
               ####
               'fibre9l_1': [234 / 255, 170 / 255, 255 / 255],
               'fibre9l_2': [255 / 255, 108 / 255, 134 / 255],
               'fibre9l_3': [255 / 255, 182 / 255, 100 / 255],
               'fibre9l_4': [180 / 255, 151 / 255, 255 / 255],
               'fibre9l_6': [248 / 255, 255 / 255, 136 / 255],
               'fibre9l_7': [136 / 255, 172 / 255, 255 / 255],
               'fibre9l_8': [133 / 255, 255 / 255, 226 / 255],
               'fibre9l_9': [135 / 255, 255 / 255, 132 / 255],
               'fibre9d_1': [95 / 255, 0 / 255, 125 / 255],
               'fibre9d_2': [157 / 255, 0 / 255, 28 / 255],
               'fibre9d_3': [155 / 255, 82 / 255, 0 / 255],
               'fibre9d_4': [40 / 255, 0 / 255, 147 / 255],
               'fibre9d_6': [119 / 255, 125 / 255, 0 / 255],
               'fibre9d_7': [0 / 255, 39 / 255, 139 / 255],
               'fibre9d_8': [0 / 255, 106 / 255, 85 / 255],
               'fibre9d_9': [53 / 255, 119 / 255, 0 / 255],
               ####
               'ggred': [217 / 255, 83 / 255, 25 / 255],
               'ggblue': [30 / 255, 144 / 255, 229 / 255],
               'ggpurple': [145 / 255, 125 / 255, 240 / 255],
               'ggyellow': [229 / 255, 220 / 255, 90 / 255],
               'gglred': [237 / 255, 103 / 255, 55 / 255],
               'gglblue': [20 / 255, 134 / 255, 209 / 255],
               'gglpurple': [165 / 255, 145 / 255, 255 / 255],
               'gglyellow': [249 / 255, 240 / 255, 110 / 255],
               'ggdred': [197 / 255, 63 / 255, 5 / 255],
               'ggdblue': [0 / 255, 94 / 255, 169 / 255],
               'ggdpurple': [125 / 255, 105 / 255, 220 / 255],
               'ggdyellow': [209 / 255, 200 / 255, 70 / 255],
               }
    return colours


# set rcParams for nice plots ################################################
def ggplot():
    colours = palette()
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.fantasy'] = 'Nimbus Mono'
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 8
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = colours['mnk_purple']
    plt.rcParams['axes.labelcolor'] = colours['mnk_yellow']
    plt.rcParams['xtick.color'] = colours['mnk_purple']
    plt.rcParams['ytick.color'] = colours['mnk_purple']
    plt.rcParams['axes.edgecolor'] = colours['mnk_lgrey']
    plt.rcParams['savefig.edgecolor'] = colours['mnk_lgrey']
    plt.rcParams['axes.facecolor'] = colours['mnk_dgrey']
    plt.rcParams['savefig.facecolor'] = colours['mnk_dgrey']
    plt.rcParams['grid.color'] = colours['mnk_lgrey']
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['axes.titlepad'] = 6


# Set up figure for plotting #################################################
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=4):
    ggplot()
    cs = palette()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1, cs


# Save 2d plot with a colourscheme suitable for ppt, as a png ################
def PPT_save_2d(fig, ax, name):

    # Set plot colours
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')

    # Loop to check for file - appends filename with _# if name already exists
    f_exist = True
    app_no = 0
    while f_exist is True:
        if os.path.exists(name + '.png') is False:
            ax.figure.savefig(name)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no))
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


# Prepare the directories and channel names ##################################
def prep_dirs_chs(d0):
    d1 = d0 + r'\time difference files'

    try:
        os.mkdir(d1)
    except OSError:
        print("Creation of the directory %s failed" % d1)
    else:
        print("Successfully created the directory %s " % d1)

    return d1


# Check if string is number ##################################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Saturation curve ###########################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


# Plot the I sat curve for data stored in file. Note plt.show() ##############
def I_sat_plot(file, directory=None, title=''):
    if directory == None:
        head, tail = os.path.split(v0)
        directory = tail

    data = np.genfromtxt(file, delimiter=',', skip_header=0)
    Ps = data[0]
    kcps = data[1] / 1000

    initial_guess = (1.5e2, 1e-1, 1e3, 1e0)
    popt, _ = opt.curve_fit(I_sat, Ps, kcps, p0=initial_guess,
                            bounds=((0, 0, 0, 0),
                                    (np.inf, np.inf, np.inf, np.inf)))

    Ps_fit = np.linspace(np.min(Ps), np.max(Ps), 1000)
    Isat_fit = I_sat(Ps_fit, *popt)

    Sat_cts = np.round(popt[0])
    P_sat = np.round(popt[1] * 1e3, 3)
    Prop_bkg = np.round(popt[2])
    bkg = np.round(popt[3])

    lb0 = 'fit'
    lb1 = 'I$_{sat}$ = ' + str(Sat_cts) + 'kcps'
    lb2 = 'bkg = ' + str(Prop_bkg) + 'P + ' + str(bkg)
    lb3 = 'I$_{sat}$ = ' + str(Sat_cts) + 'kcps, ' + \
        'P$_{sat}$ = ' + str(P_sat) + 'mW'

    ##########################################################################
    # Plot some figures
    ##########################################################################
    ax1, fig1, cs = set_figure(name='figure',
                               xaxis='Power (mW)',
                               yaxis='kcounts per secound',
                               size=4)

    plt.plot(Ps, kcps, 'o:', label='data')
    plt.plot(Ps_fit, Isat_fit, '-', label=lb0)
    plt.plot(Ps_fit, I_sat(Ps_fit, popt[0], popt[1], 0, popt[3]),
             '--', label=lb1)
    plt.plot(Ps_fit, I_sat(Ps_fit, 0, popt[1], popt[2], popt[3]),
             '--', label=lb2)

    ax1.legend(loc='lower right', fancybox=True, framealpha=1)

    plt.title(lb3)
    plt.tight_layout()
    # plt.show()
    ax1.legend(loc='lower right', fancybox=True,
               facecolor=(1.0, 1.0, 1.0, 0.0))
    os.chdir(directory)
    PPT_save_2d(fig1, ax1, 'Isat')
    plt.close(fig1)
    return Sat_cts, P_sat


# Plot the x/y tracking sweep ################################################
def plot_int_profile(directory):
    csvs = glob.glob(directory + r'\*.csv')
    x_data = np.genfromtxt(csvs[0], delimiter=',')
    x_fit = np.genfromtxt(csvs[1], delimiter=',')
    y_data = np.genfromtxt(csvs[2], delimiter=',')
    y_fit = np.genfromtxt(csvs[3], delimiter=',')

    ax1, fig1, cs = set_figure(name='figure',
                               xaxis='ΔVs',
                               yaxis='cps')
    ax1.plot(x_data[0] - np.mean(x_data[0]), x_data[1], '.',
             c=cs['ggdred'], label='x')
    ax1.plot(x_fit[0] - np.mean(x_fit[0]), x_fit[1], c=cs['gglred'])
    ax1.plot(y_data[0] - np.mean(y_data[0]), y_data[1], '.',
             c=cs['ggdblue'], label='y')
    ax1.plot(y_fit[0] - np.mean(y_fit[0]), y_fit[1], c=cs['gglblue'])

    plt.tight_layout()
    ax1.legend(loc='lower right', fancybox=True,
               facecolor=(1.0, 1.0, 1.0, 0.0))
    PPT_save_2d(fig1, ax1, 'Profiles')
    plt.close(fig1)


# Run start-stop to get dt list & save #######################################
def gen_dts_from_tts(d0, TCSPC, t_lim=100000, chA='ch0', chB='ch1'):

    # d0 is the directory containing the arrival time files

    d1 = os.path.split(d0)[0] + r"\time difference files"
    # d1 is ths directory to save the dt files to
    os.chdir(d0)

    datafiles0 = glob.glob(d0 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d0 + r'\*' + chB + r'*')

    d_dt = d1 + r"\dts " + chA + " & " + chB
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    try:
        os.mkdir(d_dt)
    except OSError:
        print("Creation of the directory %s failed" % d_dt)
    else:
        print("Successfully created the directory %s " % d_dt)

    last_file = np.min([len(datafiles0), len(datafiles1)])
    print(last_file)

    dt_chs = 'dts for chs ' + a + b
    # define a ROI range to check for co-incidences over
    dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0
    dt_file_number = 0
    for i0, v0 in enumerate(datafiles0):
        os.chdir(d_dt)
        # print output for keeping track of progress
        # print('calc', dt_chs, 'file', i0, 'of', len(datafiles0))
        tta = np.loadtxt(datafiles0[i0])
        ttb = np.loadtxt(datafiles1[i0])

        # convert to ns
        # note the conversion factor is 1e2 for HH & 1e-1 for FCT
        if TCSPC == 'HH':
            tt0 = [j0 * 1e2 for j0 in tta]
            tt1 = [j0 * 1e2 for j0 in ttb]
        elif TCSPC == 'FCT':
            tt0 = [j0 * 1e-1 for j0 in tta]
            tt1 = [j0 * 1e-1 for j0 in ttb]
        else:
            print('Choose hardware, HH or FCT')
            break

        # calc total time and # counts, count rates & gradient functions
        tot_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-9
        global_t += tot_t
        c0 = len(tt0)
        c1 = len(tt1)

        cps0 = c0 / tot_t
        cps1 = c1 / tot_t

        global_cps0.append(cps0)
        global_cps1.append(cps1)


        # calculate closest values
        dts = closest_val(tt1, tt0, dts)
        q, r = divmod(i0, 10)
        if i0 % 10 == 0:
            dt_file_number += 1
            print('saving dts', dt_file_number)
            dt_file = 'dts ' + chA + ' ' + chB + ' ' + str(dt_file_number)
            np.save(dt_file, np.asarray(dts))
            dts = []

    os.chdir(d_dt)
    dt_file = dt_file = 'dts ' + chA + ' ' + chB + ' ' + 'f'
    np.save(dt_file, np.asarray(dts))
    dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)

    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')


# Generate histogram vals and bins from dt list & save #######################
def hist_1d(d2, res=0.4, t_range=25100, chA='ch0', chB='ch1'):
    d3 = d2 + r"\dts " + chA + " & " + chB
    os.chdir(d3)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    file_str = (r'\dts ' + chA + ' ' + chB)
    print(file_str)
    datafiles = glob.glob(d3 + file_str + r'*')
    print(len(datafiles))

    nbins = int(2 * t_range / res + 1)
    edges = np.linspace(-t_range, t_range, nbins + 1)
    hists = np.zeros(nbins)

    for i0, v0 in enumerate(datafiles[0:]):
        # print('saving hist & bins csv - ', i0, 'of', len(datafiles))
        dts = np.load(v0, allow_pickle=True)
        hist, bin_edges = np.histogram(dts, edges)
        hists += hist

    np.savetxt("g2_hist.csv", hists, delimiter=",")
    np.savetxt("g2_bins.csv", bin_edges, delimiter=",")


# Plot g2 from histogram of counts ###########################################
def plot_1d_hist(d2, xlim=1000, chA='ch0', chB='ch1'):
    d3 = d2 + r"\dts " + chA + " & " + chB
    f0 = d3 + r"\g2_hist.csv"
    f1 = d3 + r"\g2_bins.csv"
    f2 = d3 + r"\other_global.csv"

    hist = np.genfromtxt(f0)
    bin_edges = np.genfromtxt(f1)

    bin_w = (bin_edges[1] - bin_edges[0]) / 2
    print('max hist value:', np.max(hist))

    ts = np.linspace(bin_edges[1], bin_edges[-1] -
                     bin_w, len(bin_edges) - 1)

    total_t, ctsps_0, ctsps_1 = np.genfromtxt(f2)
    g2s = hist / (ctsps_0 * ctsps_1 * 1e-9 * total_t * 2 * bin_w)

    print('total cps:', np.round(ctsps_0 + ctsps_1))

    ##########################################################################
    # Plot data
    ##########################################################################

    os.chdir(d3)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='cts', size=4)
    plt.title(chA + ' & ' + chB)
    ax1.set_xlim(-1 * xlim, xlim)
    ax1.set_ylim(-0.1 * np.max(hist), 1.1 * np.max(hist))

    ax1.plot(ts, hist,
             '.-', markersize=5,
             lw=0.5,
             alpha=1, label='')
    # plt.show()
    os.chdir(d2)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    plotname = 'hist ' + a + b
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)


# Function which calculates closest time differences #########################
def closest_val(a, b, dts):
    c = np.searchsorted(a, b)

    for i0, v0 in enumerate(c):
        if v0 < len(a):
            dt0 = b[i0] - a[v0 - 1]
            dt1 = b[i0] - a[v0]
            if np.abs(dt1) >= np.abs(dt0):
                dt = dt0
            else:
                dt = dt1
            dts.append(dt)
        else:
            dt0 = b[i0] - a[v0 - 1]
            dts.append(dt)
    return dts


##############################################################################
# Do some stuff
##############################################################################
d0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20200710\Sequences\10Jul20-001")
os.chdir(d0)
f = open('Summary.txt', 'w')
all_dirs = glob.glob(d0 + '\*')
peak_dirs = []
for i0, v0 in enumerate(all_dirs):
    head, tail = os.path.split(v0)
    if is_number(tail) == True:
        peak_dirs.append(v0)


for i0, v0 in enumerate(peak_dirs[64:]):
    # print(v0)
    peak_number = os.path.split(v0)[1]
    # Plot profiles
    os.chdir(v0)
    plot_int_profile(v0)
    print('###### Peak no.', peak_number, 'interation', i0, '###############')

    # Plot ISat curve
    try:
        Psat_dir = v0 + r'\PSats'
        os.chdir(Psat_dir)
        fs = glob.glob(Psat_dir + r'\*.txt')
        f0 = fs[0]
        Sat_cts, P_sat = I_sat_plot(f0, v0)
        s0 = ('Peak '+ str(peak_number) + ' Saturation counts = ' + str(Sat_cts) + 'kps')
        s1 = ('Peak '+ str(peak_number) + ' Saturation power = ' + str(P_sat) + 'mW')
        # Process and plot HBT
        if Sat_cts < 30 or P_sat > 10:
          print('not suitable for HBT')
          f.write('Peak '+ str(peak_number) + ' not suitable for HBT'+ '\n')
        else:
          d1 = prep_dirs_chs(v0)
          d0 = glob.glob(v0 + r'\HH*')[0]
          gen_dts_from_tts(d0, 'HH', 100000)
          hist_1d(d1, 10, 2000)
          plot_1d_hist(d1, 2000)
          f.write('Peak '+ str(peak_number) + ' HBT plotted'+ '\n')
    except:
        f.write('Peak '+ str(peak_number) + ' PSat data not found'+ '\n')
        pass

f.close()

##############################################################################
# Plot some figures
##############################################################################
os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################

# ax1, fig1, cs = set_figure(name='figure',
#                            xaxis='τ / ns',
#                            yaxis='probability of detection',
#                            size=4)
# ax1.plot(x, y1, c=cs['ggdred'])
# ax1.plot(x, y2, c=cs['ggdred'])
# ax1.plot(t, SPS2)
# ax1.plot(t, SPS3)
# ax1.plot(t, SPS4)
# ax1.set_ylim(-0.1, 1.1)
# plt.close(fig1)
# fig1.tight_layout()
# plt.show()

# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(x + 50, a, '.')
# plt.plot(x + 50, b, '.')
# plt.plot(c, '.')
# # plt.title()
# fig1.tight_layout()
# plt.show()


# hist/bar plot ##############################################################
# hists, bins = np.hist(δt0,100)
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
# scattter = ax3.scatter(*coords, z, '.', alpha=0.4,
#                       color=cs['gglred'], label='')
# contour = ax3.contour(*coords, z, 10, cmap=cm.jet)
# surface = ax3.plot_surface(*coords, z, 10, cmap=cm.jet)
# wirefrace = ax3.plot_wireframe(*coords, z, 10, cmap=cm.jet)
# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

# img plot ###################################################################
# ax4, fig4, cs = set.figure('image', 'x axis', 'y axis')
# im4 = plt.imshow(Z, cmap='magma', extent=extents(y) +
#                  extents(x),vmin=0,vmax=100)
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb4 = fig4.colorbar(im4, cax=cax)

# save plot ###################################################################
# ax1.figure.savefig('g2s.svg')
# plot_file_name = plot_path + 'plot2.png'
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
# PPT_save_2d(fig1, ax1, 'g2')
