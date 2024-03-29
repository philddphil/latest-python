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


##############################################################################
# Some defs
##############################################################################
#### Custom palette for plotting #############################################
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
def ggplot_sansserif():
    colours = palette()
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
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


#### Set up figure for plotting ##############################################
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=4):
    ggplot_sansserif()
    cs = palette()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1, cs


# Save 2d plot with a colourscheme suitable for ppt, as a png ################
def PPT_save_2d(fig, ax, name, dpi=600):

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
            ax.figure.savefig(name, dpi=dpi)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no), dpi=dpi)
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


##############################################################################
# Do some stuff
##############################################################################
pX = (r"C:\Data\SCM\SCM Data 20210716\Lock and log data 173716")
pY = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20210719\Lock and log data 130748")
p0 = pX

datafiles = glob.glob(p0 + r"\*.txt")
Ps = []
Is = []
ts = []
lines_total = 0

for i0, v0 in enumerate(datafiles):
    print('file name:', v0)
    with open(v0, 'r') as input_file:
        line_number = 0
        for line in input_file:
            (q, r) = divmod(line_number, 3)
            if r == 0:
                ts.append(float(line.rstrip()))
            if r == 1:
                Is.append(float(line.rstrip()))
            if r == 2:
                Ps.append(1000 * float(line.rstrip()))
            line_number += 1
        lines_total += line_number
print('Total lines', lines_total)
os.chdir(p0)
##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################

ax1, fig1, cs = set_figure(name='figure1',
                           xaxis='time / s',
                           yaxis='cts / s',
                           size=4)
ax1a = ax1.twinx()

ax1.plot(ts[0:], Is[0:], '.', c=cs['ggred'])
ax1a.plot(ts[0:], Ps[0:], '.', c=cs['ggblue'])
ax1.set_ylabel('cts/s', color=cs['ggred'])
ax1a.set_ylabel('laser power / mW', color=cs['ggblue'])

ax1.set_ylim(bottom=0)
# plt.close(fig1)
fig1.tight_layout()

ax2, fig2, cs = set_figure(name='figure2',
                           xaxis='cts / s',
                           yaxis='number',
                           size=4)
plt.hist(Is[0:], bins=50,
         facecolor=cs['ggred'],
         edgecolor=cs['mnk_dgrey'],
         alpha=0.8)


ax3, fig3, cs = set_figure(name='figure3',
                           xaxis='powers / mW',
                           yaxis='number',
                           size=4)

plt.hist(Ps[0:],
         facecolor=cs['ggblue'],
         edgecolor=cs['mnk_dgrey'],
         alpha=0.8)

plt.show()


ax1a.tick_params(axis='y', colors='xkcd:black')
PPT_save_2d(fig1, ax1, 'Laser and cts')
PPT_save_2d(fig2, ax2, 'cts hist')
PPT_save_2d(fig3, ax3, 'Laser hist')
