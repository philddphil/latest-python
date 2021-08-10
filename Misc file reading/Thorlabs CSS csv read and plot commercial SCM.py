##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=3):
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

os.chdir(r"C:\local files\Compiled Data\Confocal characterisation")

f_405 = (r"C:\local files\Compiled Data\Confocal characterisation\405nm peak.csv")
f_458 = (r"C:\local files\Compiled Data\Confocal characterisation\458nm peak.csv")
f_488 = (r"C:\local files\Compiled Data\Confocal characterisation\488nm peak.csv")
f_515 = (r"C:\local files\Compiled Data\Confocal characterisation\515nm peak.csv")
f_543 = (r"C:\local files\Compiled Data\Confocal characterisation\543nm peak.csv")
f_630 = (r"C:\local files\Compiled Data\Confocal characterisation\630nm peak.csv")

f_Wg = (r"C:\local files\Compiled Data\Confocal characterisation\Hg WG.csv")

fs = [f_405, f_458, f_488, f_515, f_543, f_630]
lbs = ['405 nm', '458 nm', '488 nm', '515 nm', '543 nm', '630 nm']
cs = ['xkcd:indigo', 'xkcd:blue', 'xkcd:teal',
      'xkcd:aquamarine', 'xkcd:green', 'xkcd:red']
ax0, fig0, cs0 = set_figure('spectra', 'wavelength / nm', 'arb.int', 5)
# for i0,v0 in enumerate(fs):
i0 = 4
	# v0 = fs[i0]

data = np.genfromtxt(f_Wg, skip_header=36, skip_footer=1, delimiter='\t')
x = data[:, 0]
y = data[:, 1]

x_new = np.linspace(x.min(), x.max(), 130000)
y_interp = interp1d(x, y, kind='quadratic')
y_smooth = y_interp(x_new)
x_max = x_new[np.argmax(y_smooth)]
# text = '    ' + str(np.round(x_max, 2))

plt.plot(data[:, 0], data[:, 1] / np.max(data[:, 1]),
         '-',
         color=cs[i0],
         label='Epi - WG')
	# plt.text(x_max, 1, text)
	# plt.plot(x_new, y_smooth / np.max(y_smooth),
	#          color='xkcd:charcoal')
ax0.set_xlim((500, 580))

ax0.legend(loc='upper right', fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.show()

ax0.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))

PPT_save_2d(fig0, ax0, '458 nm')
