##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


##############################################################################
# Some defs
##############################################################################
# Load spec file (.txt) ######################################################
def load_spec(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.readlines()
    data_line = 0
    skip_head = 0
    while skip_head == 0:
        try:
            float(data[data_line].split('\t')[0])
        except ValueError:
            data_line = data_line + 1
        else:
            skip_head = 1
    λ = []

    cts = np.zeros(shape=(len(data) - data_line,
                          len(data[data_line].split('\t')) - 1))
    for i0, val0 in enumerate(data[data_line:]):
        λ = np.append(λ, float(val0.split("\t")[0]))
        for i1, val1 in enumerate(val0.split("\t")[1:]):
            cts[i0][i1] = float(val1)

    λ = np.squeeze(np.flip(λ))
    cts = np.squeeze(np.flip(cts))

    return λ, cts


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


# Cosmic ray removal ##########################################################
def cos_ray_rem(data, thres):
    cts_diff = np.pad(np.diff(cts), 3)
    data_proc = data
    for i0, v0 in enumerate(cts_diff):
        if v0 > thres:
            print(v0)
            # if gradient of data[n] is above threshold, relace it with mean
            # of data[n-2] & data[n+2]
            data_proc[i0 - 2] = np.mean([data[i0 - 4], data[i0]])
    return data_proc



##############################################################################
# Do some stuff
##############################################################################
# Specify results directory and change working directory to this location
p0 = (r'C:\Data\SCM\20210729 Spec data')
# p0 = (r"D:\Experimental Data\Internet Thorlabs optics data"))
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)

# Initialise lists of datasets
for i0, v0 in enumerate(datafiles[-1:]):
    # load each spec file, generate λ array and cts array
    λ, cts = load_spec(v0)
   
    spec_name = (os.path.splitext(os.path.basename(v0))[0])

    idx0 = (np.abs(λ - 652.5)).argmin()
    idx1 = (np.abs(λ - 655)).argmin()
    cts0 = cts[idx1:idx0]
    λ0 = λ[idx1:idx0]

    cts1 = cos_ray_rem(cts, 400)

    ax0, fig0, cs = set_figure('Spectrum', 'wavelength / nm', 'arb. int.')
    ax0.plot(λ, cts,
             'o',
             alpha=0.5,
             color=cs['ggred'])
    ax0.plot(λ, cts1,
             '-',
             alpha=0.5,
             color=cs['ggpurple'])
    # ax0.plot(λ, gaussian_filter(cts, sigma=23),
    #     '-',
    #     color=cs['ggblue'])
    ax0.set_ylim(bottom=0)
    # ax0.set_xlim(640, 660)
    plt.tight_layout()
    plt.show()
    PPT_save_2d(fig0, ax0, spec_name)


print(np.sum(cts[idx1:idx0]))
