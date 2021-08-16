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
from scipy.optimize import curve_fit


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


# For use with extents in imshow ##############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


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
def count_hist(d3, bins, chA, TCSPC):
    print(d3)
    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')

    all_t = 0
    all_ph = 0
    all_hists = np.zeros(len(bins) - 1)
    rate = []
    ts = []

    for i0, v0 in enumerate(datafiles0[0:]):
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

            exp_t = tt[-1] - tt[0]
            exp_ph = len(tt)
            dts = np.diff(tt)
            hist, bins_alt = np.histogram(dts, bins)
            rate.append(exp_ph / (exp_t * 1e-9))
            ts.append(exp_t * 1e-9)

        all_hists += hist
        all_t += exp_t
        all_ph += exp_ph

    return all_hists, all_t, all_ph, rate, ts


# Save plot for powerpoint ###################################################
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


def Bi_exp(x, t1, t2, a1=0.5, a2=0.5):
    f1 = a1 * np.exp(-1 * x * t1)
    f2 = a2 * np.exp(-1 * x * t2)
    return f1 + f2


def Bi_exp_fit(x, y, t1, t2, a1=0.5, a2=0.5):
    init = [t1, t2, a1, a2]
    popt, pcov = curve_fit(Bi_exp,
                           x, y, p0=[*init],
                           bounds=([-np.inf, -np.inf, 0, 0],
                                   [np.inf, np.inf, 1, 1]))
    return popt, pcov


##############################################################################
# Do some stuff
##############################################################################
d0_NV = (r"C:\local files\Compiled Data\Nu Quantum\Sample 2\Example NV\HH T3")

d0_hBN = (r"C:\local files\Compiled Data\Nu Quantum\Sample 2\A2 C4 data\Peak 21\HH")

# load up NV data
d1NV = d0_NV + r"\Py data"
ch = 'ch1'
os.chdir(d1NV)
rate_NV, rate_NVts = np.loadtxt(ch + ' avg')
hist_NV = np.loadtxt(ch + ' histogram')
bin_axis_NV = np.loadtxt(ch + ' bin_axis')
arrivals_time_NV = np.loadtxt(ch + ' photon number, time')
exp_arrivals_NV = arrivals_time_NV[0]
exp_time_NV = arrivals_time_NV[1]
cpns_NV = exp_arrivals_NV / exp_time_NV
exp_decay_NV = [np.exp(-i0 * cpns_NV) for i0 in bin_axis_NV]
norm_hist_NV = [i0 / np.max(hist_NV) for i0 in hist_NV]
resids_1_NV = np.asarray(exp_decay_NV) - np.asarray(norm_hist_NV)
popt_NV, pcov_NV = Bi_exp_fit(bin_axis_NV, norm_hist_NV, cpns_NV, cpns_NV)
fit_NV = Bi_exp(bin_axis_NV, *popt_NV)
resids_2_NV = fit_NV - np.asarray(norm_hist_NV)
print(np.shape(rate_NV))
print(np.shape(rate_NVts))
d1 = d0_hBN + r"\Py data"
try:
    os.mkdir(d1)
except:
    pass

os.chdir(d0_hBN)


bins = np.linspace(0, 2e5, 401)

bin_width = bins[1] - bins[0]

bin_axis = np.linspace((bin_width) / 2, bins[-1] - (bin_width / 2),
                       len(bins) - 1)

# # Calculate dts for 1 channel, histogram & save
hist, exp_time, exp_arrivals, rate, ts = count_hist(d0_hBN, bins, ch,  'HH')
rate_ts = np.cumsum(ts)
os.chdir(d1)
np.savetxt(ch + ' photon number, time', [exp_arrivals, exp_time])
np.savetxt(ch + ' histogram', hist)
np.savetxt(ch + ' bin_axis', bin_axis)
np.savetxt(ch + ' avg', [rate, rate_ts])

# Load previous datasets
arrivals_time = np.loadtxt(ch + ' photon number, time')

exp_arrivals = arrivals_time[0]
exp_time = arrivals_time[1]

hist = np.loadtxt(ch + ' histogram')
bin_axis = np.loadtxt(ch + ' bin_axis')
rate, rate_ts = np.loadtxt(ch + ' avg')

norm_hist = [i0 / np.max(hist) for i0 in hist]

cpns = exp_arrivals / exp_time

exp_decay = [np.exp(-i0 * cpns) for i0 in bin_axis]

popt, pcov = Bi_exp_fit(bin_axis, norm_hist, cpns, cpns)
print(exp_arrivals)
print(exp_time)
print('rate = ', cpns * 1e9)
print('t1 = ', np.round(popt[0] * 1e9))
print('t2 = ', np.round(popt[1] * 1e9))
print('a1 = ', np.round(popt[2], 2))
print('a2 = ', np.round(popt[3], 2))

fit = Bi_exp(bin_axis, *popt)

resids_1 = np.asarray(exp_decay) - np.asarray(norm_hist)
resids_2 = fit - np.asarray(norm_hist)

# rate = [1e9 / i0 for i0 in dts]
# avg0 = rate[::100]
# avg1 = rate[::1000]
# avg2 = rate[::10000]
# avg3 = rate[::100000]
##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################
xlabel = 'Δt (ns) - bin width ' + str(bin_width) + 'ns'
ax0, fig0, cs = set_figure(name='figure0',
                           xaxis=xlabel,
                           yaxis='#',
                           size=4)
plt.plot(bin_axis, hist / np.max(hist), '.',
         label='hBN data')
plt.plot(bin_axis, exp_decay, c=cs['ggdred'],
         label='y = e^{-t/a}')
plt.plot(bin_axis_NV, hist_NV / np.max(hist_NV), '.', c=cs['ggblue'],
         label='NV data')
plt.plot(bin_axis_NV, exp_decay_NV, c=cs['ggdblue'],
         label='y = e^{-t/a}')

# plt.plot(bin_axis, Bi_exp(bin_axis, *popt))
fig0.tight_layout()

# ax1, fig1, cs = set_figure(name='figure1',
#                            xaxis=xlabel,
#                            yaxis='log(#)',
#                            size=4)
# plt.bar(bin_axis, hist, bin_width)
# plt.plot(bin_axis, exp_decay, c=cs['ggblue'])
ax0.set_yscale('log')
fig0.tight_layout()

ax2, fig2, cs = set_figure(name='figure2',
                           xaxis='t',
                           yaxis='average cps (over 5 s)',
                           size=4)
plt.plot(rate_ts, rate,'.-',
         label='hBN',
         lw=0.5,
         markersize=7,
         alpha=1)

plt.plot(rate_NVts, rate_NV,'.-',
         label='NV',
         lw=0.5,
         markersize=7,
         alpha=1)

ax2.set_ylim(0, np.max([1.1 * np.max(rate), 1.1 * np.max(rate_NV)]))
fig2.tight_layout()

ax3, fig3, cs = set_figure(name='figure3',
                           xaxis=xlabel,
                           yaxis='#',
                           size=4)
plt.plot(bin_axis, resids_1,
         label='$y = e^{-t/a}$')
plt.plot(bin_axis, resids_2,
         label='$y = e^{-t/a}+e^{-t/b}$',
         c=cs['ggdred'])
plt.plot(bin_axis_NV, resids_1_NV,
         label='$y = e^{-t/a}$')
plt.plot(bin_axis_NV, resids_2_NV,
         c=cs['ggdblue'],
         label='$y = e^{-t/a}+e^{-t/b}$',)
fig3.tight_layout()
plt.show()
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
ax0.legend(loc='upper right', fancybox=True, framealpha=0.0)
ax2.legend(loc='upper right', fancybox=True, framealpha=0.0)
ax3.legend(loc='upper right', fancybox=True, framealpha=0.0)
PPT_save_2d(fig0, ax0, 'hist')
# PPT_save_2d(fig1, ax1, 'log hist')
PPT_save_2d(fig2, ax2, 'avg')
PPT_save_2d(fig3, ax3, 'resids')
