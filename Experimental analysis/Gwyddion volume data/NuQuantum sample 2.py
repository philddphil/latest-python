##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
from itertools import permutations
from itertools import combinations
from scipy.ndimage.filters import uniform_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# Prepare the directories and channel names ##################################
def prep_dirs(d0):
    d1 = d0 + r'\py data'
    d2 = d1 + r'\time difference files'
    d3 = d1 + r"\arrival time files"

    try:
        os.mkdir(d1)
    except:
        pass
    try:
        os.mkdir(d2)
    except:
        pass
    try:
        os.mkdir(d3)
    except:
        pass

    return d1, d2, d3


# Calculate the running average count rate in chA for last N photons
def count_rate(d3, chA, N, TCSPC):
    print(d3)
    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    dts_avg = []
    crs_avg = []
    for i0, v0 in enumerate(datafiles0[0:2]):
        print(v0)

        if TCSPC == 'HH':
            TT = np.loadtxt(datafiles0[i0])
            data_loop = enumerate([0])

        if TCSPC == 'FCT':
            TT = np.load(datafiles0[i0], allow_pickle=True)
            data_loop = enumerate(TTs[0])

        for i1, v1 in data_loop:
                 # convert to ns
            # note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt = [j0 * 1e2 for j0 in TT]

            elif TCSPC == 'FCT':
                tt = [j0 * 1e-1 for j0 in TT[i1]]

            else:
                print('Choose hardware, HH or FCT')
                break

            dt = np.diff(tt)
            cr = 1 / (dt * 1e-9)
            cr_avg = np.convolve(cr, np.ones((N,)) / N, mode='valid')
            cr_avg0 = uniform_filter1d(cr, N)
            crs_avg.append(cr_avg)

    return crs_avg


# Find regions/objects in image
def image_object_find(x_img, y_img, img, u_lim):
    y_img = y_img[::-1]
    krnl_size = 10
    x_k = np.arange(krnl_size)
    y_k = np.arange(krnl_size)

    coords = np.meshgrid(x_k, y_k)

    G = Gaussian_2D(coords, 1, int(krnl_size - 1) / 2,
                    int(krnl_size - 1) / 2, 1, 1)
    G = np.reshape(G, (krnl_size, krnl_size))

    img_1 = ndi.convolve(img, G)

    img_2 = img_1
    super_threshold_indices = img_1 < u_lim
    img_1[super_threshold_indices] = 0

    img_3, label_nmbr = ndi.label(img_2)

    peak_slices = ndi.find_objects(img_3)
    centroids_px = []
    for peak_slice in peak_slices:
        dy, dx = peak_slice
        x, y = dx.start, dy.start
        cx, cy = centroid(img_3[peak_slice])
        centroids_px.append((x + cx, y + cy))

    return centroids_px, peak_slices


# Generic 2D Gaussian peak function ###########################################
def Gaussian_2D(coords, A, x_c, y_c, σ_x, σ_y, θ=0, bkg=0, N=1):
    x, y = coords
    x_c = float(x_c)
    y_c = float(y_c)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    G = (bkg + A * np.exp(- (a * ((x - x_c) ** 2) +
                             2 * b * (x - x_c) * (y - y_c) +
                             c * ((y - y_c) ** 2))**N))
    return G.ravel()


# find the central value for rectangular region w x h of data array ##########
def centroid(data):
    h, w = np.shape(data)
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)

    cx = np.sum(X * data) / np.sum(data)
    cy = np.sum(Y * data) / np.sum(data)

    return cx, cy


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
# Do some stuff
##############################################################################
d0 = (r"C:\local files\Experimental Data\G4 L12 Rennishaw\20201020")

f0 = (r"C:\local files\Experimental Data\G4 L12 Rennishaw\20201020\One z per line.txt")
f1 = (r"C:\local files\Experimental Data\G4 L12 Rennishaw\20201020\wavelengths.txt")
f2 = (r"C:\local files\Experimental Data\G4 L12 Rennishaw\20201020\mean image.txt")

mean_image_data = np.genfromtxt(f2)
img = mean_image_data

log_img = np.log(mean_image_data)

y_px, x_px = np.shape(mean_image_data)
x_range = 56.2
y_range = 54.2
x_img = np.linspace(0, x_range, x_px)
y_img = np.linspace(0, y_range, y_px)
y_m = y_img[1] - y_img[0]
x_m = x_img[1] - x_img[0]
y_c = np.min(y_img)
x_c = np.min(x_img)

u_lim = 15

d1_name = str(u_lim) + ' thresh'
d1 = os.path.join(d0, d1_name) 

clim = [0, 10]
centroids_px, peak_slices = image_object_find(
    x_img, y_img, mean_image_data, u_lim)
print('Objects found (#) = ', len(centroids_px))

wavelengths_data = np.genfromtxt(f1, skip_header=3)
wavelengths_micron = [1e9 * pair[1] for pair in wavelengths_data]

# find regions and all the spectra contained within
specs_all = []
for i0, v0 in enumerate(peak_slices[0:]):
  specs_object = []
  indices = []
  yslice, xslice = v0
  x_range = np.arange(xslice.start, xslice.stop)
  y_range = np.arange(yslice.start, yslice.stop)
  X, Y = meshgrid(x_range,y_range)
  for i1, v1 in enumerate(x_range):
    for i2, v2 in enumerate(y_range):
      index = x_px * v2 + v1
      indices.append(index)
  print(i0, len(indices)) 
  with open(f0, 'r') as input_file:
    for position, line in enumerate(input_file):
      if position in indices:
        data_raw = line
        data_split = re.split(r'\t+', line)
        data_floats = [float(value) for value in data_split]
        specs_object.append(data_floats)
  specs_all.append(specs_object)

print(np.shape(specs_all[0]))
print(np.shape(specs_all[1]))
##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################

ax1, fig1, cs = set_figure('fig1', size=7)
im1 = plt.imshow(np.log(img), cmap='magma',
                 vmin=np.min(np.log(img)),
                 vmax=0.9 * np.max(np.log(img)))

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax)
cbar1.ax.get_yaxis().labelpad = 15
cbar1.set_label('log counts / second', rotation=270)


for i0, v0 in enumerate(centroids_px):
    x, y = v0
    ax1.plot(x, y, 'o',
             ms=15,
             mec=cs['mnk_green'],
             fillstyle='none')
    ax1.text(x, y, '  ' + str(i0),
             c=cs['mnk_green'])

ax2, fig2, cs = set_figure('fig2',  size=7)
im2 = plt.imshow(img, cmap='magma',
                 vmin=clim[0] + 1, vmax=0.5 * clim[1]
                 )
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig2.colorbar(im2, cax=cax)
cbar2.ax.get_yaxis().labelpad = 15
cbar2.set_label('counts / second', rotation=270)

for i0, v0 in enumerate(centroids_px[0:]):
    x, y = v0
    ax2.plot(x, y,
             'o',
             ms=15,
             mec=cs['mnk_green'],
             fillstyle='none')
    ax2.text(x, y,
             '  ' + str(i0),
             c=cs['mnk_green'])
 
    os.chdir(d1)
    ax3, fig3, cs = set_figure('fig3', '', '', size=4)
    specs_object = specs_all[i0]
    
    for i1, v1 in enumerate(specs_object):
      plt.plot(wavelengths_micron, v1)

    plt.tight_layout()
    name = 'obj' + str(i0) + ' x' + str(int(x)) + ' y' + str(int(y))
    PPT_save_2d(fig3, ax3, name)
    plt.close()


# hist/bar plot ##############################################################
# hists, bins = np.hist(δt0,100)
# size = 9
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Country', fontsize=28, labelpad=80,)
# ax2.set_ylabel('Money (M$)', fontsize=28)
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


# save plot ###################################################################
plt.show()

os.chdir(d0)
ax1.figure.savefig('mean log counts.svg')
ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
ax2.figure.savefig('mean counts.svg')
ax2.legend(loc='upper left', fancybox=True, framealpha=0.0)

cbar1.set_label('log counts / second', rotation=270, color='xkcd:black')
cbar2.set_label('counts / second', rotation=270, color='xkcd:black')

PPT_save_2d_im(fig1, ax1, cbar1, 'mean log counts')
# PPT_save_2d_im(fig2, ax2, cbar2, 'mean counts')
# PPT_save_2d(fig3, ax3, 'spec')
