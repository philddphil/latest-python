##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy import ndimage as ndi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
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


# For use with extents in imshow ##############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# Load SCM image #############################################################
def load_SCM_F5L10(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'X initial' in j0:
            x_init = float(data[i0].split("\t")[-1])
        if 'X final' in j0:
            x_fin = float(data[i0].split("\t")[-1])
        if 'X res' in j0:
            x_res = float(data[i0].split("\t")[-1])
        if 'Y initial' in j0:
            y_init = float(data[i0].split("\t")[-1])
        if 'Y final' in j0:
            y_fin = float(data[i0].split("\t")[-1])
        if 'Y res' in j0:
            y_res = float(data[i0].split("\t")[-1])
        if 'y V' in j0:
            data_start_line = i0 + 2

    x = np.linspace(x_init, x_fin, int(x_res))
    y = np.linspace(y_fin, y_init, int(y_res))
    img = np.loadtxt(data[data_start_line:])
    return (x, y, img)


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


# Find 'objects' (centro-symmetric features) in an image #####################
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
    super_threshold_indices = img_1 < Ulim
    img_1[super_threshold_indices] = 0

    img_3, label_nmbr = ndi.label(img_2)

    peak_slices = ndi.find_objects(img_3)
    print('Objects found (#) = ', len(peak_slices))
    centroids_px = []
    for peak_slice in peak_slices:
        dy, dx = peak_slice
        x, y = dx.start, dy.start
        cx, cy = centroid(img_3[peak_slice])
        centroids_px.append((x + cx, y + cy))

    shape_img = np.shape(img)
    x_px = np.arange(0, shape_img[0])
    y_px = np.arange(0, shape_img[1])

    x_px_m = (x_px[-1] - x_px[0]) / len(x_px)
    y_px_m = (y_px[-1] - y_px[0]) / len(y_px)

    x_img_m = (x_img[-1] - x_img[0]) / len(x_img)
    y_img_m = (y_img[-1] - y_img[0]) / len(y_img)
    centroids_img = []

    for x, y in centroids_px:
        centroids_img.append(((x * (x_img_m / x_px_m) - (x_px[0] - x_img[0])),
                              (y * (y_img_m / y_px_m) - (y_px[0] - y_img[0]))))

    return centroids_img


##############################################################################
# File paths
##############################################################################

p0 = (r"C:\Data\SCM\SCM Data 20210721\Raster scans\21Jul21 scan-006.txt")
d0 = (r"C:\Data\SCM\SCM Data 20210721\Raster scans")

p0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Examplar Data\Confoal images\WSe2 Bilayer\27Aug21 scan-001.txt")
d0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Examplar Data\Confoal images\WSe2 Bilayer")

##############################################################################
# Image processing to retrieve peak locations
##############################################################################
Ulim = 10000000
clim = [20000, 1500000]

x_img, y_img, img0 = load_SCM_F5L10(p0)
img = np.flipud(img0)
im_min = np.min(np.min(img))
im_mean = np.mean(img)
centroids_img = image_object_find(x_img, y_img, img, Ulim)
coords_save = image_object_find(x_img, y_img, img0, Ulim)


##############################################################################
# Plot some figures
##############################################################################

size = 4
ax1, fig1, cs = set_figure(name='figure log',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=size
                               )

im1 = plt.imshow(img, cmap='magma',
                     extent=extents(x_img) + extents(y_img),
                     norm=LogNorm(vmin=im_mean/5, vmax=np.max(img)),
                     origin='upper'
                     )


ax1, fig1, cs = set_figure('fig1', size=4)
im1 = plt.imshow(np.log(img), cmap='magma',
                 extent=extents(x_img) + extents(y_img),
                 vmin=np.min(np.log(img)),
                 vmax=0.9 * np.max(np.log(img))
                 )


divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig1.colorbar(im1, cax=cax)
cbar1.ax.get_yaxis().labelpad = 15

cbar1.set_label('counts / second', rotation=270)

cbar1.set_label('counts / second', rotation=270, c='xkcd:black')


pk_number = 0
for x, y in centroids_img:
    pk_number += 1
    ax1.plot(x, y, 'o',
             ms=15,
             mec=cs['mnk_green'],
             fillstyle='none')
    ax1.text(x, y, '    ' + str(pk_number),
             c=cs['mnk_green'])



ax2, fig2, cs = set_figure(name='figure lin',
                               xaxis='x distance (μm)',
                               yaxis='y distance (μm)',
                               size=size
                               )

ax2, fig2, cs = set_figure('fig2',  size=3)

im2 = plt.imshow(img, cmap='magma',
                 extent=extents(x_img) + extents(y_img),
                 vmin=clim[0], vmax=clim[1]
                 )
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax)
cbar2.ax.get_yaxis().labelpad = 15
cbar2.set_label('counts / second', rotation=270, c='xkcd:black')
pk_number = 0
for x, y in centroids_img:
    pk_number += 1
    ax2.plot(x, y, 'o',
             ms=15,
             mec=cs['mnk_green'],
             fillstyle='none')
    ax2.text(x, y, '    ' + str(pk_number),
             c=cs['mnk_green'])

plt.tight_layout()
# ax3, fig3, cs = set_figure('fig3')
# im3 = plt.imshow(img_2, cmap='magma',
#                  extent=extents(y) +
#                  extents(x),
#                  vmin=clim[0], vmax = clim[1]
#                  )
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig3.colorbar(im3, cax=cax)

# ax4, fig4, cs = set_figure('fig4')
# im4 = plt.imshow(img_3, cmap='magma',
#                  extent=extents(y) +
#                  extents(x),
#                  # vmin=clim[0], vmax = clim[1]
#                  )
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig4.colorbar(im4, cax=cax)

plt.show()
os.chdir(d0)
np.savetxt("coords.csv", coords_save, delimiter=",")
PPT_save_2d_im(fig2, ax2, cbar2, 'Labelled image lin.png')
PPT_save_2d_im(fig1, ax1, cbar1, 'Labelled image log.png')
ax2.figure.savefig('Labelled image.svg')
