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


# Generate histogram vals and bins from dt list & save #######################
def hist_1d_fname(d2, res=0.4, t_range=25100):

    g2_f = ("range" + str(int(t_range)) +
            "ns res" + str(int(res * 1e3)) + "ps g2.csv")
    g2_hist_f = ("range" + str(int(t_range)) +
                 "ns res" + str(int(res * 1e3)) + "ps g2_hist.csv")
    bins_f = ("range" + str(int(t_range)) +
              "ns res" + str(int(res * 1e3)) + "ps g2_bins.csv")
    return g2_f, bins_f, g2_hist_f


# Generate histogram vals and bins from dt list & save #######################
def hist_2d_fname(d2, res=0.4, t_range=25100):

    g3_f = ("range" + str(int(t_range)) +
            "ns res" + str(int(res * 1e3)) + "ps g3.csv")
    g3_hist_f = ("range" + str(int(t_range)) +
                 "ns res" + str(int(res * 1e3)) + "ps g3_hist.csv")
    xbins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_xbins.csv")
    ybins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_ybins.csv")
    return g3_f, xbins_f, ybins_f, g3_hist_f


##############################################################################
# Code to do stuff
##############################################################################
# specify data directory
d0 = (r"C:\local files\Compiled Data\G3s\Single")
d1 = d0 + r"\alt results"
d2 = d0 + r"\results"
os.chdir(d1)

g2_res = 0.1
g3_res = 30 * g2_res
t_range = 1500
t_lim = 1500

g2_fa, bins_fa, g2_hist_fa = hist_1d_fname(d1, g2_res, t_range)
g2_f, bins_f, g2_hist_f = hist_1d_fname(d2, g2_res, t_range)

bins = np.genfromtxt(bins_f)
g2_hista = np.genfromtxt(g2_hist_fa)
os.chdir(d2)
g2_hist = np.genfromtxt(g2_hist_f)

x_w = bins[1] - bins[0]
ts = np.arange(bins[0] + x_w / 2, bins[-1], x_w)

set_figure()
plt.plot(ts, g2_hista / (2 * 7200))
plt.plot(10 * ts[0:len(g2_hist)] + 100, g2_hist)
plt.show()
