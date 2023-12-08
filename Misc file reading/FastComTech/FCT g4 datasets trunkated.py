
##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt


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
               'gggrey': [118 / 255, 118 / 255, 118 / 255],
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
    fig1.tight_layout()
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


# Find clock resets ###########################################################
def find_clk_resets(a):
    b = []
    for i0, v0 in enumerate(a):
        if v0 < 0:
            b.append(int(i0))
    return b


# Find clock resets ###########################################################
def find_dt_data_clk(data, clock):
    dts = []
    for i0, v0 in enumerate(data):
        t0 = clk[int(np.floor(data[i0] / 10000))]
        t1 = clk[int(np.ceil(data[i0] / 10000))]
        dts.append(data[i0] - min([t0, t1]))
    return dts


##############################################################################
# Do some stuff
##############################################################################
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211")
f1 = d0 + r'\1a.txt'
f2 = d0 + r'\2a.txt'
f3 = d0 + r'\3a.txt'
f4 = d0 + r'\4a.txt'
f6 = d0 + r'\6a.txt'

data6 = np.genfromtxt(f6, delimiter=',')
data1 = np.genfromtxt(f1, delimiter=',') - np.min(data6)
data2 = np.genfromtxt(f2, delimiter=',') - np.min(data6)
data3 = np.genfromtxt(f3, delimiter=',') - np.min(data6)
data4 = np.genfromtxt(f4, delimiter=',') - np.min(data6)

print(len(data4))

clk = (data6 - np.min(data6))
clk_y = 0 * data6
ch1_y = np.ones(len(data1))
ch2_y = 2 * np.ones(len(data2))
ch3_y = 3 * np.ones(len(data3))
ch4_y = 4 * np.ones(len(data4))

dts1 = find_dt_data_clk(data1, clk)
dts1 = np.asarray(dts1) / 10
dts2 = find_dt_data_clk(data2, clk)
dts2 = np.asarray(dts2) / 10
dts3 = find_dt_data_clk(data3, clk)
dts3 = np.asarray(dts3) / 10
dts4 = find_dt_data_clk(data4, clk)
dts4 = np.asarray(dts4) / 10
##############################################################################
# Plot some figures
##############################################################################

# xy plot ####################################################################
ax0, fig0, cs = set_figure(
    'fig0', 't / ns', 'channel #')
ax0.plot(data1, ch1_y, '.')
ax0.plot(data2, ch2_y, '.')
ax0.plot(data3, ch3_y, '.')
ax0.plot(data4, ch4_y, '.')
ax0.plot(clk, clk_y, '.')


# hist plot ##################################################################
bins = np.linspace(0, 1000, 30)
ax1, fig1, cs = set_figure('fig1', 'dt / ns', '#')
ax1.hist(dts1, bins, edgecolor=cs['mnk_dgrey'], facecolor=cs['ggred'])
ax1.set_xlim(0, 1000)
ax2, fig2, cs = set_figure('fig2', 'dt / ns', '#')
ax2.hist(dts2, bins, edgecolor=cs['mnk_dgrey'], facecolor=cs['ggblue'])
ax2.set_xlim(0, 1000)
ax3, fig3, cs = set_figure('fig3', 'dt / ns', '#')
ax3.hist(dts3, bins, edgecolor=cs['mnk_dgrey'], facecolor=cs['ggpurple'])
ax3.set_xlim(0, 1000)
ax4, fig4, cs = set_figure('fig4', 'dt / ns', '#')
ax4.hist(dts4, bins, edgecolor=cs['mnk_dgrey'], facecolor=cs['gggrey'])
ax4.set_xlim(0, 1000)
plt.show()


# # save plot ################################################################
os.chdir(d0)
PPT_save_2d(fig0, ax0, 'plot')
PPT_save_2d(fig1, ax1, 'plot')
PPT_save_2d(fig2, ax2, 'plot')
PPT_save_2d(fig3, ax3, 'plot')
PPT_save_2d(fig4, ax4, 'plot')
