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
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_file_import
import prd_plots
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Confocal measurements\SCM Data 20190404"
      r"\Raster scans")

datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
print(datafiles)
for i0, val0 in enumerate(datafiles[0:]):
    x, y, img = prd_file_import.load_SCM_F5L10(val0)
    lb = os.path.basename(val0)
    plotname = os.path.splitext(lb)[0] + '.png'
    print(lb)

    fig1 = plt.figure('fig1', figsize=(4, 4))
    prd_plots.ggplot()
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('x dimension (V)')
    ax1.set_ylabel('y dimension (V)')
    plt.title(lb)
    im1 = plt.imshow(np.flipud(img), cmap='magma',
                     extent=prd_plots.extents(y) +
                     prd_plots.extents(x),
                     label=lb,
                     vmin=np.min(img),
                     vmax=0.6 * np.max(img))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(im1, cax=cax)
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    prd_plots.PPT_save_2d(fig1, ax1, plotname)
