##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# For use with extents in imshow #############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# Load SCM image #############################################################
def load_SCM_F5L10(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'X initial / V' in j0:
            x_init = float(data[i0].split("\t")[-1])
        if 'X final / V' in j0:
            x_fin = float(data[i0].split("\t")[-1])
        if 'X increment / V' in j0:
            x_res = float(data[i0].split("\t")[-1])
        if 'Y initial / V' in j0:
            y_init = float(data[i0].split("\t")[-1])
        if 'Y final / V' in j0:
            y_fin = float(data[i0].split("\t")[-1])
        if 'Y increment / V' in j0:
            y_res = float(data[i0].split("\t")[-1])
        if 'Y wait period / ms' in j0:
            data_start_line = i0 + 2

    x = np.linspace(x_init, x_fin, int(x_res))
    y = np.linspace(y_fin, y_init, int(y_res))
    img = np.loadtxt(data[data_start_line:])
    return (x, y, img)


# Save 2d image with a colourscheme suitable for ppt, as a png ###############
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


##############################################################################
# Def some functions
##############################################################################
pX = (r"C:\Data\SCM\SCM Data 20200928\Raster scans")
pY = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20200908\Raster scans")

p0 = pX

datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
print(len(datafiles),'images found')
size=5
for i0, v0 in enumerate(datafiles[-2:]):
    print(os.path.split(v0)[1])
    
    x, y, img = load_SCM_F5L10(v0)
    img = img
    log_img = np.log(img)
    # FSM scaling: 12.5 microns = 1.56
    x = x * 25 / (2.6 - 1.4)
    y = y * 25 / (2.6 - 1.4)
    # Piezo scaling 10V = 25 microns
    # x = x * 2.5
    # y = y * 2.5

    lb = os.path.basename(v0)
    plotname1 = os.path.splitext(lb)[0]
    plotname2 = os.path.splitext(lb)[0] + ' log'
    print(plotname1)
    print(plotname2)
    ax1, fig1, cs = set_figure(name='figure lin',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=5)
    im1 = plt.imshow(np.flipud(img), cmap='magma',
                     extent=extents(y) +
                     extents(x),
                     label=lb,
                     # vmin=np.min(img),
                     vmax=1e-0 * np.max(img)
                     )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig1.colorbar(im1, cax=cax)
    cbar1.ax.get_yaxis().labelpad = 15
    cbar1.set_label('counts / second', rotation=270)
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    # ax1.figure.savefig(plotname1 + 'dark.svg')
    # ax1.figure.savefig(plotname1 + 'dark.png')
    PPT_save_2d_im(fig1, ax1, cbar1, plotname1)

    # plt.axis('off')
    # plt.cla()
    # im1 = plt.imshow(np.flipud(img), cmap='magma',
    #                  extent=prd_plots.extents(y) +
    #                  prd_plots.extents(x),
    #                  label=lb,
    #                  vmin=np.min(img),
    #                  vmax=0.6 * np.max(img),
    #                  alpha=0.5)
    # plt.savefig(plotname2)

    ax2, fig2, cs = set_figure(name='figure log',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=5)
    # plt.title(lb)
    im2 = plt.imshow(np.flipud(log_img), cmap='magma',
                     extent=extents(y) +
                     extents(x),
                     label=lb,
                     vmin=np.min(log_img),
                     vmax=1 * np.max(log_img)
                     )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(im2, cax=cax)
    cbar2 = fig2.colorbar(im2, cax=cax)
    cbar2.ax.get_yaxis().labelpad = 15
    cbar2.set_label('log [counts / second]', rotation=270)
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    PPT_save_2d_im(fig2, ax2, cbar1, plotname2)
##############################################################################
