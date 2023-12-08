##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import numpy as np
import glob as glob
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
import prd_maths
import prd_file_import
import prd_plots
np.set_printoptions(suppress=True)
cs = prd_plots.palette()

# Load HH file (.txt) ########################################################
def load_HH(filepath):
    a = open(filepath, 'r', encoding='ascii')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if '#channels per curve' in j0:
            channels = float(data[i0 + 1])
        if '#ns/bin' in j0:
            τ_res0 = float(data[i0 + 1].split("\t")[0])
        if '#counts' in j0:
            data_start_line = i0 + 1
    print(τ_res0, channels)
    τs = np.linspace(0, τ_res0 * (channels - 1), int(channels))
    all_cts = np.loadtxt(data[data_start_line:])
    cts = all_cts[:, 0]
    return (τs, cts)

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


##############################################################################
# Specify directory with the .dat files for plotting
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 HydraHarp"
      r"\HH Data 20200730")

# change current working directory to path above
os.chdir(p0)

# Generate list of files with .dat extension in directory
datafiles = glob.glob(p0 + r'\*.dat')

##############################################################################
# Directory loop so everythin is done in a for loop
##############################################################################
for i0, v0 in enumerate(datafiles[0:]):

    # create label from filename
    lb = os.path.basename(v0)
    lb = os.path.splitext(lb)[0]

    # import x/y data from filename (v0)
    ts, cts = load_HH(v0)
    
    # set plot parameters
    ggplot()
    size = 4

    # begin populating figure
    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(111)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Δt (ns)')
    ax2.set_ylabel('counts')
    t_shift = 93
    plt.plot(ts - t_shift, cts, lw=0.5,
             alpha=0.4, color=cs['ggdred'], label=lb)
    plt.plot(ts - t_shift, cts, 'o', lw=0.5,
             alpha=1, color=cs['gglred'], markersize=2)
    plt.xlim((-t_shift, 500))
    plt.ylim((0, np.max(cts)))
    plt.tight_layout()

    # show figure (pop up)
    plt.show()

    # generate filename
    plot_file_name = lb

    # save figure
    prd_plots.PPT_save_2d(fig2, ax2, 'plot')
