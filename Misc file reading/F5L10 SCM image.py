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
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_file_import
import prd_plots
cs = prd_plots.palette()

##############################################################################
# Do some stuff
p0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements\SCM Data 20191126\Raster scans")


datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
print(datafiles)
for i0, val0 in enumerate(datafiles[0:]):
    x, y, img = prd_file_import.load_SCM_F5L10(val0)
    img = img
    log_img = np.log(img)
    # FSM scaling: 12.5 microns = 1.56
    x = x * 25 / (2.6 - 1.4)
    y = y * 25 / (2.6 - 1.4)
    # Piezo scaling 10V = 25 microns
    # x = x * 2.5
    # y = y * 2.5

    lb = os.path.basename(val0)
    plotname1 = os.path.splitext(lb)[0]
    plotname2 = os.path.splitext(lb)[0] + ' log'
    print(plotname1)
    print(plotname2)
    size = 5
    fig1 = plt.figure('fig1', figsize=(size, size))
    prd_plots.ggplot()
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('x distance (μm)')
    ax1.set_ylabel('y distance (μm)')
    plt.title('')
    im1 = plt.imshow(np.flipud(img), cmap='magma',
                     extent=prd_plots.extents(y) +
                     prd_plots.extents(x),
                     label=lb,
                     # vmin=np.min(img),
                     vmax=1e-2 * np.max(img)
                     )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig1.colorbar(im1, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('counts / second', rotation=270)
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    ax1.figure.savefig(plotname1 + 'dark.svg')
    ax1.figure.savefig(plotname1 + 'dark.png')
    prd_plots.PPT_save_2d(fig1, ax1, plotname1)

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

    fig2 = plt.figure('fig2', figsize=(size, size))
    prd_plots.ggplot()
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('x distance (μm)')
    ax2.set_ylabel('y distance (μm)')
    # plt.title(lb)
    im2 = plt.imshow(np.flipud(log_img), cmap='magma',
                     extent=prd_plots.extents(y) +
                     prd_plots.extents(x),
                     label=lb,
                     vmin=np.min(log_img),
                     vmax=1 * np.max(log_img)
                     )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(im2, cax=cax)
    cbar = fig2.colorbar(im2, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar = cbar.set_label('log [counts / second]', rotation=270)
    plt.tight_layout()
    plt.show()
    ax2.figure.savefig(plotname2 + 'dark.png')
    ax2.figure.savefig(plotname2 + 'dark.svg')
    os.chdir(p0)
    prd_plots.PPT_save_2d(fig2, ax2, plotname2)
##############################################################################
