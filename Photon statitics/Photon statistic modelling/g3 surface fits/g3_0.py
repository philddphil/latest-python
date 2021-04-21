
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
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################################################
# Some defs
##############################################################################
# Modokai palette for plotting ###############################################
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


# set rcParams for nice plots #################################################
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


# Extents is used for plotting images (2D plots) with accurate axis ##########
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


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


# Save 3d plot with a colourscheme suitable for ppt, as a png #################
def PPT_save_3d(fig, ax, name):
    plt.rcParams['text.color'] = 'xkcd:black'
    fig.patch.set_facecolor('xkcd:white')
    ax.patch.set_facecolor('xkcd:white')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.zaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    ax.tick_params(axis='z', colors='xkcd:black')
    fig.savefig(name)


# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope and multiple emitters ######################
def g2_fit(t):
    a = 0.09
    b = 0.008
    c = 1.5
    d = 0.0001
    dt = 0
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    g3 = (g1 + g2)
    g4 = g3 * np.exp(-d * np.abs(t - dt))
    return g4


# Exponential lifetime function ###############################################
def Exp_decay(t, a=0.1):
    P = 1 - np.exp(-a * np.abs(t))
    return P


# Coherent c.w. source ########################################################
def CW_coh(t):
    P = np.ones(np.shape(t))
    return P


# Stepped perfect single photon source ########################################
def PSPS(t, tau=10):
    P = np.zeros(np.shape(t))
    print(np.shape(P))
    for i0, v0 in enumerate(np.arange(np.shape(t)[0])):
        for i1, v1 in enumerate(np.arange(np.shape(t)[1])):
            if np.abs(t[i0, i1]) >= tau:
                P[i0, i1] = 1
    return P


# Stepped perfect single photon source ########################################
def Double_PSPS(t1, t2, tau=10):
    P = np.ones(np.shape(t1))
    print(np.shape(P))
    for i0, v0 in enumerate(np.arange(np.shape(t1)[0])):
        for i1, v1 in enumerate(np.arange(np.shape(t1)[1])):
            if np.abs(t1[i0, i1]) <= tau and np.abs(t2[i0, i1]) <= tau:
                print(np.abs(t1[i0, i1]))
                print(np.abs(t2[i0, i1]))
                P[i0, i1] = 0

    return P


# Experimental 2d g3 curve fit ################################################
def g3_2d(P_t, coords):
    t1, t2 = coords
    g1 = P_t(t1)
    g2 = P_t(t2)
    g3 = P_t(t2 - t1)

    return g1, g2, g3


# Experimental 2d g3 curve fit ################################################
def g3_2d_alt(coords, a):
    t1, t2 = coords
    g1 = 1 - np.exp(-a * np.sqrt(np.abs(t1**2 + t2**2)))
    return g1


###############################################################################
# Do some stuff
###############################################################################
ts = np.linspace(-1501, 1501, 501)
coords = np.meshgrid(ts, ts)

z = 1
x = 1 - z

perm_sum = z * z * z  + \
    z * z * x  + \
    z * z * x  + \
    z * z * x  + \
    z * x * x  + \
    z * x * x + \
    z * x * x  + \
    x * x * x


g1, g2, g3 = g3_2d(g2_fit, coords)
f1, f2, f3 = g3_2d(g2_fit, coords)
f1 = CW_coh(coords[0])
f2 = f1
f3 = f1


g3_plot = z * z * z * (g1 * g2 * g3) + \
    z * z * x * g1 + \
    z * z * x * g2 + \
    z * z * x * g3 + \
    z * x * x * f1 + \
    z * x * x * f2 + \
    z * x * x * f3 + \
    x * x * x * (f1 * f2 * f3)

###############################################################################
# Plot some figures
###############################################################################
os.chdir(r"C:\local files\Python\Plots")
# xy plot #####################################################################
# ax1, fig1, cs = set_figure('profiles', 'τ / ns', 'g$^2$')
# ax1.plot(g3_0[-1, :], color=cs['ggdblue'])
# ax1.plot(t1s, g3_1[0, :],)
# ax1.plot(t1s, g3_1[:, -1])
# ax1.plot(ts, np.diagonal(np.fliplr(g3_plot)))
# ax1.plot(ts, g3_plot[50])
# ax1.set_ylim(-0.1, 1.1)

# img plot ####################################################################
# ax3, fig3, cs = set_figure('img plot 0', 'τ1 / ns', 'τ2 / ns')
# im3 = plt.imshow(np.flipud(g3_0), cmap='magma',
#                  extent=extents(t1s) + extents(t2s))
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb = fig3.colorbar(im3, cax=cax)

ax4, fig4, cs = set_figure('img plot 1', 'τ1 / ns', 'τ2 / ns')
im4 = plt.imshow(g3_plot, cmap='magma',
                 extent=extents(ts) + extents(ts), 
                 origin='lower',vmin=0,vmax=np.max(g3_plot))
ax4.plot(ts, -ts)
ax4.plot(ts, ts[50] * np.ones(len(ts)))
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig4.colorbar(im4, cax=cax)

# xyz plot ####################################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# ax3.contour(*coords, g3_1, 50, cmap='magma')
# scattter = ax3.plot(ts, -ts, np.diagonal(np.fliplr(g3_plot)),
#                     color=cs['ggred'], label='')
# scattter = ax3.plot(ts, ts[50] * np.ones(len(ts)), g3_plot[50],
#                     color=cs['ggblue'], label='')
# ax3.plot_surface(*coords, g3_plot, cmap='magma', alpha=0.8)
# norm = plt.Normalize(g3_plot.min(), g3_plot.max())
# colors = cm.magma(norm(g3_1_a))
# surf = ax3.plot_surface(*coords_a, g3_1_a, facecolors=colors, shade=False)
# surf.set_facecolor((0, 0, 0, 0))
# ax3.plot_wireframe(*coords_a, g3_1_a, color=cs['ggred'], lw=0.5)
# ax3.legend(loc='upper right', fancybox=True, framealpha=0.1)
# os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.set_zlim(0, np.max(g3_plot))
# ax3.view_init(elev=75, azim=-110)
# save plot ###################################################################
plt.show()
# ax2.figure.savefig('funding' + '.png')
# plot_file_name = plot_path + 'plot2.png'
# PPT_save_2d_im(fig4, ax4, cb, 'g3')
# PPT_save_2d(fig1, ax1, 'g3 profiles')
# PPT_save_3d(fig3, ax3, '3d plot')
