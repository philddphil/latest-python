##############################################################################
# Import some libraries
##############################################################################
import os
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

from itertools import permutations
from itertools import combinations


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


# Check if string is number
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


##############################################################################
# Do some stuff
##############################################################################
p0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\py data\arrival time files 0\ch1 data0"
p1 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\py data\arrival time files\ch1 f.npy"
p2 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\py data\arrival time files 0 npy\ch1 slice 1.npy"
p3 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\TEST.lst"
p4 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\ch2.txt"

data0 = np.genfromtxt(p0)
data1 = np.load(p1, allow_pickle=True)
data2 = np.load(p2, allow_pickle=True)

# f0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0\py data\time difference files\dts ch0 & ch1\dts ch0 ch1 0.npy"
# dt0 = np.load(f0, allow_pickle=True)

# tot_lines = sum(1 for line in open(p3))
# print('Total lines in .lst = ', tot_lines)


tot_lines_ch1 = sum(1 for line in open(p4))
print('Total lines in ch1 = ', tot_lines_ch1)

tot_lines_ch1_uw = 0
for i0, v0 in enumerate(data1):
  tot_lines_ch1_uw += len(data1[i0])
print('Total unwrapped lines in ch1 = ', tot_lines_ch1_uw)
print('Total resets in ch1', len(data1))

print('data0', len(data0))
print(data0[0:5])
print('data1', len(data1))
print(data1[0][0:5])
print('data2', len(data2))
print(data2[0][0:5])
##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################

# ax1, fig1, cs = set_figure(name='figure',
#                            xaxis='wavelengths (λ) / nm',
#                            yaxis='a.u',
#                            size=4)
# ax1.plot(wavelengths_nm, I, c=cs['ggred'])
# ax1.plot(x, y2, c=cs['ggdred'])

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
# ax1.figure.savefig('spec.svg')
# plot_file_name = plot_path + 'plot2.png'
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
# PPT_save_2d(fig1, ax1, 'spec')
