##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20181025"
      r"\Raster scans")
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)


for i0, val0 in enumerate(datafiles[0:]):
    x, y, img = prd.load_SCM_F5L10(val0)
    lb = os.path.basename(val0)
    plotname = os.path.splitext(lb)[0] + '.png'
    print(lb)

    fig1 = plt.figure('fig1', figsize=(4, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('x dimension (V)')
    ax1.set_ylabel('y dimension (V)')
    plt.imshow(img, extent=prd.extents(x) + prd.extents(y), label=lb,
               vmin=50, vmax=1000)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    os.chdir(p0)
    ax1.legend(loc='upper right', fancybox=True, framealpha=1)
    prd.PPT_save_2d(fig1, ax1, plotname)
