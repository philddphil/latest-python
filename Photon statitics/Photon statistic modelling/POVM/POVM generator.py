
##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
from itertools import permutations


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


# Poissonian distribution at values of k for mean value λ #####################
def Poissonian_1D(k, λ):
    P = []

    for i0, j0 in enumerate(k):
        P.append(np.exp(-λ) * (λ**j0) / sp.math.gamma(j0 + 1))

    return P


def POVM_N(k, eta, P_dc):
    N = ((1 - eta)**k) * (1 - P_dc)
    return N


def POVM_C(k, eta, P_dc):
    C = 1 - POVM_N(k, eta, P_dc)
    return C


def POVM_x(x):
    a = POVM_C
    b = POVM_C
    c = POVM_C
    d = POVM_C

    if x[0] == 0:
        a = POVM_N
    if x[1] == 0:
        b = POVM_N
    if x[2] == 0:
        c = POVM_N
    if x[3] == 0:
        d = POVM_N
    return(a, b, c, d)


def POVM_g(m, x, eta, P_dc):
    a, b, c, d = POVM_x(x)
    g_tot = 0
    for i0 in np.arange(m + 1):
        for i1 in np.arange(m + 1):
            for i2 in np.arange(m + 1):
                if i0 + i1 + i2 <= m:
                    numerator = (np.math.factorial(m)
                                 * a(i0, eta, P_dc)
                                 * b(i1, eta, P_dc)
                                 * c(i2, eta, P_dc)
                                 * d(m - i0 - i1 - i2, eta, P_dc))

                    denominator = ((4 ** m) *
                                   np.math.factorial(i0) *
                                   np.math.factorial(i1) *
                                   np.math.factorial(i2) *
                                   np.math.factorial(m - i0 - i1 - i2))

                    g = numerator / denominator
                    g_tot += g
    return g_tot


def POVM_xi(n, m, eta=1, P_dc=0):
    x0 = np.zeros(4 - n)
    x1 = np.ones(n)
    x = np.append(x0, x1)
    X = list(set(permutations(x)))
    print(X)
    xi = 0
    for i0, v0 in enumerate(X):
        g = POVM_g(m, v0, eta, P_dc)
        xi += g
    return xi


def POVM_Xi(n, m, eta=1, P_dc=0):
    Xi = np.zeros((n + 1, m + 1))
    for i0 in np.arange(n + 1):
        for i1 in np.arange(m + 1):
            print('n =', i0, 'm =', i1)
            Xi[i0, i1] = POVM_xi(i0, i1, eta)
    return Xi


##########################################################################
# Do some stuff
##########################################################################

print(POVM_Xi(4, 4))

##############################################################################
# Plot some figures
##############################################################################
# prep colour scheme for plots and paths to save figs to
# ggplot()
# cs = palette()

# # NPL path
# plot_path = r"C:\local files\Python\Plots"
# # Surface Pro path
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
# os.chdir(plot_path)

# xy plot ####################################################################

# ax1, fig1, cs = set_figure(
#     'BB radiation', 'wavelength / nm', 'Spectral radiance / W / sr m$^3$')
# ax1.plot(n, rho, '.', label=('η = ' + str(eta)))
# ax1.plot(ns, Psi, '-', lw=0.5, label=('Ψ = ' + str(mus)))
# # ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
# ax1.set_yscale('log')
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
# ax2.set_ylabel('Money (M$)', fontsize=28
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
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4,
#                       color=cs['gglred'], label='')
# surffit = ax3.contour(*coords, z, 10, cmap=cm.jet)
# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

# img plot ###################################################################
# size = 4
# fig4 = plt.figure('fig4', figsize=(size * np.sqrt(2), size))
# ax4 = fig4.add_subplot(1, 1, 1)
# fig4.patch.set_facecolor(cs['mnk_dgrey'])
# ax4.set_xlabel('x dimension')
# ax4.set_ylabel('y dimension')
# plt.title('')
# im4 = plt.imshow(Z, cmap='magma', extent=extents(y) +
#                  extents(x))
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig4.colorbar(im4, cax=cax)

# save plot ###################################################################
# ax2.figure.savefig('funding' + '.png')
# plot_file_name = plot_path + 'plot2.png'
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
# PPT_save_2d(fig1, ax1, 'BB radiation')
