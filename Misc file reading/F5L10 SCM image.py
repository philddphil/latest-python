##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


##############################################################################
# Def some functions
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


# For use with extents in imshow #############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# Load SCM image #############################################################
def load_SCM_F5L10(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()

#### New file reading structure
    for i0, j0 in enumerate(data):
      if 'X initial' in j0:
          x_init = float(data[i0].split("\t")[-1])
      if 'X final' in j0:
          x_fin = float(data[i0].split("\t")[-1])
      if 'X res' in j0:
          x_res = float(data[i0].split("\t")[-1])
          print(x_res)
      if 'Y initial' in j0:
          y_init = float(data[i0].split("\t")[-1])
      if 'Y final' in j0:
          y_fin = float(data[i0].split("\t")[-1])
      if 'Y res' in j0:
          y_res = float(data[i0].split("\t")[-1])
      if 'Y wait period / ms' in j0:
          data_start_line = i0 + 2

#### Old file reading structure

    # for i0, j0 in enumerate(data):
    #   if 'X initial' in j0:
    #       x_init = float(data[i0].split("\t")[-1])
    #   if 'X final' in j0:
    #       x_fin = float(data[i0].split("\t")[-1])
    #   if 'X increment' in j0:
    #       x_res = float(data[i0].split("\t")[-1])
    #       print(x_res)
    #   if 'Y initial' in j0:
    #       y_init = float(data[i0].split("\t")[-1])
    #   if 'Y final' in j0:
    #       y_fin = float(data[i0].split("\t")[-1])
    #   if 'Y increment' in j0:
    #       y_res = float(data[i0].split("\t")[-1])
    #   if 'Y wait period / ms' in j0:
    #       data_start_line = i0 + 2

    x = np.linspace(x_init, x_fin, int(x_res))
    y = np.linspace(y_fin, y_init, int(y_res))
    img = np.loadtxt(data[data_start_line:])
    return (x, y, img)


# Save 2d image with a colourscheme suitable for ppt, as a png ###############
def PPT_save_2d_im(fig, ax, cb, name, dpi=600):
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
    cbylabel_obj = plt.getp(cb.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='xkcd:black')

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
# Def some functions
##############################################################################

##### PXI file path
pX = (r"C:\Data\SCM\SCM Data 20210621\Raster scans")
##### Office laptop file paths
pY = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20210513\Raster scans")
pZ = (r"C:\local files\Compiled Data\Nu Quantum"
      r"\Sample 2\B2 C1 data\Raster scans")
p0 = pX

datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
print(len(datafiles), 'images found')
for i0, v0 in enumerate(datafiles):
    print(i0, v0)

size = 3
for i0, v0 in enumerate(datafiles[:]):
    print(os.path.split(v0)[1])

    x, y, img = load_SCM_F5L10(v0)
    im_min = np.min(np.min(img))
    im_mean = np.mean(img)
    print('image min value =', im_min)
    #### Use scaling if reading older files (second set of ifs in 'load_SCM_F5L10') 
    #### Scaling for old files with voltages saved, not microns
    #### FSM scaling: 12.5 microns = 1.56
    # x = x * 25 / (2.6 - 1.4)
    # y = y * 25 / (2.6 - 1.4)
    #### Piezo scaling 10V = 25 microns
    # x = x * 2.5
    # y = y * 2.5

    # print(np.min(img))
    lb = os.path.basename(v0)
    plotname1 = os.path.splitext(lb)[0]
    plotname2 = os.path.splitext(lb)[0] + ' log'
    print(plotname1)
    print(plotname2)
    ax1, fig1, cs = set_figure(name='figure lin',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=size)
    im1 = plt.imshow(img, cmap='magma',
                     extent=extents(y) +  extents(x),
                     label=lb,
                     vmin=np.min(img),
                     vmax=0.5 * np.max(img),
                     origin='lower'
                     )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig1.colorbar(im1, cax=cax)
    cbar1.ax.get_yaxis().labelpad = 15
    cbar1.set_label('counts / second', rotation=270, c='xkcd:black')
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    # ax1.figure.savefig(plotname1 + 'dark.svg')
    # ax1.figure.savefig(plotname1 + 'dark.png')
    PPT_save_2d_im(fig1, ax1, cbar1, plotname1)

    ax2, fig2, cs = set_figure(name='figure log',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=size
                               )

    im2 = plt.imshow(img, cmap='magma',
                     extent=extents(y) + extents(x),
                     norm=LogNorm(vmin=im_mean/5, vmax=np.max(img)),
                     label=lb,
                     origin='lower'
                     )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig2.colorbar(im2, cax=cax)
    cbar2 = fig2.colorbar(im2, cax=cax)
    cbar2.ax.get_yaxis().labelpad = 15
    cbar2.set_label('log [counts / second]', rotation=270, c='xkcd:black')
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    PPT_save_2d_im(fig2, ax2, cbar2, plotname2)
##############################################################################
