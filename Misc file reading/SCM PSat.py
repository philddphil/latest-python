##############################################################################
# Import some libraries
##############################################################################
import os
import glob
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


# set rcParams for nice plots ################################################
def ggplot_sansserif():
    colours = palette()
    # plt.style.use('ggplot')
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


# Set up figure for plotting #################################################
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



# Saturation curve ###########################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


# Plot the I sat curve for data stored in file. Note plt.show() ##############
def I_sat_plot(file, title=''):
    data = np.genfromtxt(file, delimiter=',', skip_header=0)
    Ps = 1000*data[0]
    kcps = data[1] / 1000

    initial_guess = (1.5e2, 1, 0, 0)
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
    # PPT_save_2d(fig1, ax1, 'Psat1')
    plt.close(fig1)
    return Ps, kcps, popt

##############################################################################
# Do some stuff
##############################################################################
d0 = (r'C:\local files\Compiled Data\G3s\Other data\PSats')
fs = glob.glob(d0 + r'\*.txt')
f0 = fs[0]
f1 = fs[1]
os.chdir(d0)
Ps_0, kcps_0, popt_0 = I_sat_plot(f0)
Ps_1, kcps_1, popt_1 = I_sat_plot(f1)

Ps_fit = np.linspace(np.min(Ps_0), np.max(Ps_0), 1000)

Isat_fit_0 = I_sat(Ps_fit, *popt_0)
Isat_fit_1 = I_sat(Ps_fit, *popt_1)

ax1, fig1, cs = set_figure(name='figure',
                           xaxis='Power (mW)',
                           yaxis='$10^3$ counts per secound',
                           size=2.5)

plt.plot(Ps_0, kcps_0, '.', label='data',
  color=cs['ggdblue'])
plt.plot(Ps_fit, Isat_fit_0, '-',
  color=cs['gglblue'])

plt.plot(Ps_1, kcps_1, '.', 
  label='data',
  color=cs['ggdred']
  )
plt.plot(Ps_fit, Isat_fit_1, '-',
  color=cs['gglred'])

plt.tight_layout()
plt.show()
PPT_save_2d(fig1, ax1, 'Plot')