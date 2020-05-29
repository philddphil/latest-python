##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
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


# Saturation curve ############################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


##############################################################################
# Do some stuff
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements") +\
    (r"\SCM Data 20200309\PSats")
f0 = r"\10Mar20-001 - Peak3_Isat_1.txt"

data = np.genfromtxt(p0 + f0, delimiter=',', skip_header=0)

with open(p0 + f0) as f:
    content = f.readlines()
peakxy = content[0].rstrip()
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


##############################################################################
# Plot some figures
##############################################################################
size = 2.5
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ggplot()
cs = palette()

ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Inferred power (mW)')
ax2.set_ylabel('kcounts per secound')
plt.plot(Ps, kcps, 'o:', label='data')
plt.plot(Ps_fit, Isat_fit, '-', label=lb0)
plt.plot(Ps_fit, I_sat(Ps_fit, popt[0], popt[1], 0, popt[3]),
         '--', label=lb1)
plt.plot(Ps_fit, I_sat(Ps_fit, 0, popt[1], popt[2], popt[3]),
         '--', label=lb2)

ax2.legend(loc='lower right', fancybox=True, framealpha=1)

plt.title(lb3)
plt.tight_layout()
plt.show()
os.chdir(p0)
ax2.legend(loc='lower right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
PPT_save_2d(fig2, ax2, 'peak3 0.png')
