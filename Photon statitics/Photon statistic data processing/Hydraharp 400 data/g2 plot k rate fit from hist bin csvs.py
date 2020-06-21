##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


###############################################################################
# Defs
###############################################################################
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


# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope ############################################
def g2_3_lvl_exp(t, dt, a, b, c, d, bkg):
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    h = (g1 + g2) * np.exp(-d * np.abs(t - dt)) + bkg
    return h


# k rate using identities from Berthel et al 2015
def k_rate_3lvl(t, dt, k12, k21, k31, k23, R, bkg):
    a = k12 + k21
    b = k31 + (k12 * k23) / (k12 + k21)
    c = 1 + (k12 * k23) / (k31 * (k12 + k21))

    # P = k31 / (k31 - k21 + (k21 + k23) * (1 + k31 / k12))
    # T = R / (k21 * P)

    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    h = (g1 + g2) * np.exp(- R * np.abs(t - dt)) + bkg
    return h

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20200310\HH T3 142814\t range 100000 i2 range 0")
f0 = p0 + r"\4ns_res_hist.csv"
f1 = p0 + r"\4ns_res_bins.csv"
f2 = p0 + r"\other_global.csv"

hist = np.genfromtxt(f0)
bin_edges = np.genfromtxt(f1)

bin_w = (bin_edges[1] - bin_edges[0]) / 2

ts = np.linspace(bin_edges[1] +
                 bin_w, bin_edges[-1] -
                 bin_w, len(bin_edges) - 1)

total_t, ctsps_0, ctsps_1 = np.genfromtxt(f2)
g2s = hist / (ctsps_0 * ctsps_1 * 1e-9 * total_t * 2 * bin_w)

print('total cps = ', np.round(ctsps_0 + ctsps_1))

##############################################################################
# Fit data
##############################################################################
# fit curve using exponential fits defined in g2_3_lvl_exp
a = 0.5
b = 0.002
c = 1.5
d = 0.00001
dt = 0
bkg = 0.2

init_g = [dt, a, b, c, d, bkg]
popt_g, pcov_g = curve_fit(g2_3_lvl_exp,
                           ts, g2s, p0=[*init_g],
                           bounds=([-np.inf, 0, 0, 0, 0, 0],
                                   [np.inf, np.inf, np.inf,
                                    np.inf, np.inf, np.inf]))

rnd = 5
print('dt =', np.round(popt_g[0], rnd),
      'a =', np.round(popt_g[1], rnd),
      'b =', np.round(popt_g[2], rnd),
      'c =', np.round(popt_g[3], rnd),
      'd =', np.round(popt_g[4], rnd),
      'bgk =', np.round(popt_g[5], rnd))

# fit curve using k rate fit (over fitted)
# note lambda denotes anonymous function to lock R in fit
k12_i = 0.25
k21_i = 0.05696
k23_i = 0.1
k31_i = 0.01
dt_i = 0
bkg_i = 0.2
# calc total count rate in ns-1
R = (ctsps_0 + ctsps_1) / 1e9

init_k = [dt_i, k12_i, k31_i, k23_i, bkg_i]

(dt, k12, k31, k23, bkgk), pcov_k = curve_fit(lambda t, dt, k12, k31, k23, bkg:
                                              k_rate_3lvl(
                                                  t, dt, k12, k21_i,
                                                  k31, k23, R, bkg),
                                              ts, g2s,
                                              p0=[*init_k],
                                              bounds=([-np.inf, 0, 0, 0, 0],
                                                      [np.inf, np.inf, np.inf,
                                                       np.inf, np.inf]))

popt_kp = [dt, k12, k21_i, k31, k23, R, bkgk]

print('dt =', np.round(popt_kp[0], rnd),
      'k12 =', np.round(popt_kp[1], rnd),
      'k21 =', np.round(popt_kp[2], rnd),
      'k31 =', np.round(popt_kp[3], rnd),
      'k23 =', np.round(popt_kp[4], rnd),
      'R =', np.round(popt_kp[5], rnd),
      'bgk =', np.round(popt_kp[6], rnd))

P = k31 / (k31 - k21_i + (k21_i + k23) * (1 + k31 / k12))
T = R / (k21_i * P)
print(T)

# print('lifetime = ', np.round(1 / popt_k[2], 3), ' ns')
init_kp = [dt, k12, k21_i, k31, k23, R, bkg]
##############################################################################
# Plot data
##############################################################################
ggplot()
cs = palette()
ts_fit = np.linspace(ts[0], ts[-1], 5000000)
# # os.chdir(plot_path)

# xy plot ####################################################################
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('τ, ns')
ax1.set_ylabel('g2s')
# ax1.set_ylim(0, 1.1 * np.max(hist))

ax1.plot(bin_edges[1:], g2s,
         '.', markersize=5,
         lw=0.5,
         alpha=1, label='')
# ax1.plot(ts_fit, g2_3_lvl_exp(
#     ts_fit, *popt_g), '-',
#     color=cs['ggblue'],
#     label='Fit',
#     lw=1.0)
ax1.plot(ts_fit, k_rate_3lvl(
    ts_fit, *popt_kp), '-',
    color=cs['mnk_green'],
    label='Fit',
    lw=1.0)
# ax1.plot(ts_fit, np.exp(-np.abs(ts_fit * R)), '-',
#          color=cs['mnk_yellow'],
#          label='Fit',
#          lw=1.0)

plt.show()
plotname = 'g2s'
PPT_save_2d(fig1, ax1, plotname)
