##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from scipy.ndimage.filters import uniform_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nptdms import TdmsFile
from nptdms import TdmsFile


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


# For use with extents in imshow #############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# Prepare the directories and channel names ##################################
def prep_dirs(d0):
    d1 = d0 + r'\py data'
    d2 = d1 + r'\time difference files'
    d3 = d1 + r"\arrival time files"

    try:
        os.mkdir(d1)
    except:
        pass
    try:
        os.mkdir(d2)
    except:
        pass
    try:
        os.mkdir(d3)
    except:
        pass

    return d1, d2, d3


# Calculate the running average count rate in chA for last N photons
def count_rate(d3, chA, N, TCSPC):
    print(d3)
    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    dts_avg = []
    crs_avg = []
    for i0, v0 in enumerate(datafiles0[0:2]):
        print(v0)

        if TCSPC == 'HH':
            TT = np.loadtxt(datafiles0[i0])
            data_loop = enumerate([0])

        if TCSPC == 'FCT':
            TT = np.load(datafiles0[i0], allow_pickle=True)
            data_loop = enumerate(TTs[0])

        for i1, v1 in data_loop:
                 # convert to ns
            # note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt = [j0 * 1e2 for j0 in TT]

            elif TCSPC == 'FCT':
                tt = [j0 * 1e-1 for j0 in TT[i1]]

            else:
                print('Choose hardware, HH or FCT')
                break

            dt = np.diff(tt)
            cr = 1 / (dt * 1e-9)
            cr_avg = np.convolve(cr, np.ones((N,)) / N, mode='valid')
            cr_avg0 = uniform_filter1d(cr, N)
            crs_avg.append(cr_avg)

    return crs_avg


# Load Agilent Verbose XY ascii ###############################################
def load_AgilentDCA_ascii(filepath):
    a = open(filepath, 'r', encoding='ascii')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'Points' in j0:
            Points = float(data[i0].split(":")[-1])
        if 'Count' in j0:
            Count = float(data[i0].split(":")[-1])
        if 'XInc' in j0:
            XInc = float(data[i0].split(":")[-1])
        if 'XOrg' in j0:
            XOrg = float(data[i0].split(":")[-1])
        if 'YData range' in j0:
            YData_range = float(data[i0].split(":")[-1])
        if 'YData center' in j0:
            YData_center = float(data[i0].split(":")[-1])
        if 'Coupling' in j0:
            Coupling = data[i0].split(":")[-1]
        if 'XRange' in j0:
            XRange = float(data[i0].split(":")[-1])
        if 'XOffset' in j0:
            XOffset = float(data[i0].split(":")[-1])
        if 'YRange' in j0:
            YRange = float(data[i0].split(":")[-1])
        if 'YOffset' in j0:
            YOffset = float(data[i0].split(":")[-1])
        if 'Date' in j0:
            Date = data[i0].split(":")[-1]
        if 'Time' in j0:
            Time = data[i0].split(":")[-1]
        if 'Frame' in j0:
            Frame = data[i0].split(":")[-1]
        if 'X Units' in j0:
            X_unit = data[i0].split(":")[-1]
        if 'Y Units' in j0:
            Y_unit = data[i0].split(":")[-1]
        if 'XY Data' in j0:
            data_start_line = i0 + 1

    XY_scope_data = np.loadtxt(data[data_start_line:], delimiter=',')
    return (XY_scope_data, YOffset)


# Save 2d plot with a colourscheme suitable for ppt, as a png #################
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
    dpi = 600
    while f_exist is True:
        if os.path.exists(name + '.png') is False:
            ax.figure.savefig(name,  dpi=dpi)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no), dpi=dpi)
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


# Generic 1D Lorentzian function ##############################################
def Lorentzian_1D(x, A, x_c, γ, bkg=0):
    L = (A * γ ** 2) / ((x - x_c)**2 + γ ** 2) + bkg
    return L
##############################################################################
# Do some stuff
##############################################################################
c = 3e8
h = 6.626e-34
e = 1.602e-19
λs = 1e-9 * np.linspace(400, 900, 1000)
Δλ = 5e-9
λc = 550e-9
Es = (h * c) / (λs * e)
ΔE = Δλ * ((h * c) / (e * λc**2))

L1=Lorentzian_1D(Es, 1, 2.25426, ΔE)
L2=Lorentzian_1D(λs, 1, 550e-9, Δλ)
##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################
ax1, fig1, cs=set_figure(name='figure',
                           xaxis='Wavelength (nm)',
                           yaxis='Amplitude',
                           size=4)

ax1.plot(λs, L1, label='Lor in E')
ax1.plot(λs, L2, label='Lor in λ')

fig1.tight_layout()
plt.show()
# os.chdir(d0)
# ax1.figure.savefig('IM output.png')
# PPT_save_2d(fig1, ax1, 'IM output.png')

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
# ax4, fig4, cs = set_figure('lin image', 'x axis', 'y axis')
# im4 = plt.imshow(mean_image_data, cmap='magma')
# im4 = plt.imshow(mean_image_data, cmap='magma',
#                  extent=extents(y) + extents(x),
#                  vmin=0,
#                  vmax=10)
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb4 = fig4.colorbar(im4, cax=cax)


# ax5, fig5, cs = set_figure('log image', 'x axis', 'y axis')
# im5 = plt.imshow(log_img, cmap='magma',
#                  extent=extents(y) +
#                  extents(x),
#                  vmin=np.min(log_img),
#                  vmax=1 * np.max(log_img))
# divider = make_axes_locatable(ax5)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig5.colorbar(im5, cax=cax)
# cb5 = fig5.colorbar(im5, cax=cax)
# cb5.ax.get_yaxis().labelpad = 15
# cb5.set_label('log [counts / second]', rotation=270)
# plt.tight_layout()
# plt.show()

# save plot ###################################################################
ax1.figure.savefig('spec.svg')
plot_file_name = 'plot2.png'
ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
PPT_save_2d(fig1, ax1, 'spec')
