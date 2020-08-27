
##############################################################################
# Import some libraries
##############################################################################
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################################################
# Some defs
##############################################################################
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


# Save 2d image with a colourscheme suitable for ppt, as a png ################
def PPT_save_2d_im(fig, ax, cb, name):
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
    # cbylabel_obj = plt.getp(cb.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='xkcd:black')

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


# For use with extents in imshow ##############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# 2d histogram the arrival time pairs in the directory d2
def hists_2d_csvs(d2, res=10, t_range=25010, chA='ch0', chB='ch1', chC='ch2'):
    d4 = (d2 + r"\dts " + chA + " & " + chB +
          " and " + chA + " & " + chC)
    os.chdir(d4)
    datafiles = glob.glob(d4 + r'\dts*')

    nbins = int(2 * t_range / res) + 1
    x_edges = np.linspace(-t_range, t_range, nbins + 1)
    y_edges = np.linspace(-t_range, t_range, nbins + 1)
    dts = []

    data = list(csv.reader(open(datafiles[0])))
    dt1s = []
    dt2s = []
    hists = np.zeros((nbins, nbins))

    hist = []

    for i0, v0 in enumerate(datafiles[0:]):
        data = list(csv.reader(open(datafiles[i0])))
        dt1s = []
        dt2s = []
        print("dts " + chA + " & " + chB + " and " + chA + " & " + chC,
              '- saving hist & bins csv -', i0, 'of',
              len(datafiles), 'max', np.max(hists))
        for i1, v1 in enumerate(data):

            if len(v1) == 2:
                dt1s.append(int(float(v1[0])))
                dt2s.append(int(float(v1[1])))
        dt1s_a = np.asarray(dt1s)
        dt2s_a = np.asarray(dt2s)
        # dt1s_int = dt1s_a.astype(np.int32)
        # dt2s_int = dt2s_a.astype(np.int32)
        hist, xbins, ybins = np.histogram2d(dt1s_a, dt2s_a, [x_edges, y_edges])
        hists = hists + hist

    hist_csv_name = ("g3_hist_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    xbins_csv_name = ("g3_xbins_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    ybins_csv_name = ("g3_ybins_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    np.savetxt(hist_csv_name, hists, delimiter=",")
    np.savetxt(xbins_csv_name, x_edges, delimiter=",")
    np.savetxt(ybins_csv_name, y_edges, delimiter=",")
    return hists, x_edges, y_edges


# Prepare the directories and channel names
def prep_dirs_chs(d0):
    d1 = d0 + r'\py data'
    d2 = d1 + r'\time difference files'
    d3 = d1 + r"\arrival time files"

    try:
        os.mkdir(d1)
    except OSError:
        print("Creation of the directory %s failed" % d1)
    else:
        print("Successfully created the directory %s " % d1)

    try:
        os.mkdir(d2)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    try:
        os.mkdir(d3)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    Chs = ['ch0', 'ch1', 'ch2', 'ch3']
    Chs_combs = list(set(combinations(Chs, 3)))
    return d1, d2, d3, Chs_combs


# Plot a 2d histogram from data in d2
def plot_1set_2d_hist(d4, res=10, t_range=25010):
    hist_csv_name = ("g3_hist_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    xbins_csv_name = ("g3_xbins_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    ybins_csv_name = ("g3_ybins_res_" + str(res) +
                     "_range_" + str(t_range) + ".csv")
    os.chdir(d4)

    hist = np.loadtxt(hist_csv_name, delimiter=',')
    x_edges = np.genfromtxt(xbins_csv_name)
    y_edges = np.genfromtxt(ybins_csv_name)

    bin_w = (x_edges[1] - x_edges[0]) / 2

    ts = np.linspace(x_edges[0] +
                     bin_w, x_edges[-1] -
                     bin_w, len(x_edges) - 1)

    # total_t, ctsps_0, ctsps_1, ctsps_2 = np.genfromtxt(f3)
    # print('time =', total_t)
    # print('cts 0 =', ctsps_0)
    # print('cts 1 =', ctsps_1)
    # print('cts 2 =', ctsps_2)
    # print('bin_w =', 2 * bin_w)

    # # normalise the Glauber function
    # g3s = hist / (ctsps_2 * ctsps_0 * ctsps_1 * 4 * total_t *
    #               (2 * bin_w * 1e-9) ** 2)
    # g3s = g3s
    # print('total cps = ', np.round(ctsps_0 + ctsps_1 + ctsps_2))

    ##########################################################################
    # Plot data
    ##########################################################################

    # profile plots ##########################################################
    hist_x, hist_y = np.shape(hist)
    ax1, fig1, cs = set_figure('profiles', 'time', 'g^3')
    hist_0 = (np.shape(hist)[0] - 1) / 2
    print(hist_0)
    ax1.plot(ts, hist[:, 20], '.--')
    ax1.plot(ts, np.diag(np.fliplr(hist)), '.--')

    # img plot ###############################################################
    ax4, fig4, cs = set_figure('image', 'x axis', 'y axis')
    ax4.plot(ts, np.ones(len(ts)) * ts[20], lw=1)
    ax4.plot(ts, -ts, lw=1)
    im4 = plt.imshow(hist, cmap='magma',
                     extent=extents(x_edges) + extents(y_edges),
                     vmin=0, vmax=np.max(hist) - 50, origin='lower')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig4.colorbar(im4, cax=cax)
    # plt.show()
    print('max counts', np.max(hist))
    os.chdir(d4)
    PPT_save_2d(fig1, ax1, 'profiles')
    PPT_save_2d_im(fig4, ax4, cb, 'image')
    plt.close(fig1)
    plt.close(fig4)


##############################################################################
# Load the data and histogram values
##############################################################################
d0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200721\1800"

d1, d2, d3, Chs_combs = prep_dirs_chs(d0)
t_range = 100000
res = 500
nbins = int(2 * t_range / res) + 1
all_hists = np.zeros((nbins, nbins))

for i0, v0 in enumerate(Chs_combs):
    chA, chB, chC = v0[0:3]
    d4 = (d2 + r"\dts " + chA + " & " + chB +
          " and " + chA + " & " + chC)
    hists, xbins, ybins = hists_2d_csvs(d2, res, t_range, chA, chB, chC)
    plot_1set_2d_hist(d4, res, t_range)
    all_hists += hists

os.chdir(d0)
hist_csv_name = ("g3_hist_res_" + str(res) +
                 "_range_" + str(t_range) + ".csv")
xbins_csv_name = ("g3_xbins_res_" + str(res) +
                 "_range_" + str(t_range) + ".csv")
ybins_csv_name = ("g3_ybins_res_" + str(res) +
                 "_range_" + str(t_range) + ".csv")
np.savetxt(hist_csv_name, all_hists, delimiter=",")
np.savetxt(xbins_csv_name, xbins, delimiter=",")
np.savetxt(ybins_csv_name, ybins, delimiter=",")

plot_1set_2d_hist(d0, res, t_range)
