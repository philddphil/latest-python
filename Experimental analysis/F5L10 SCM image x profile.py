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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20181010"
      r"\Raster scans")
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)
y_px = 34

for i0, val0 in enumerate(datafiles[1:2]):
    x, y, img = prd.load_SCM_F5L10(val0)
    lb = os.path.basename(val0)
    print(lb, np.shape(img))

    img_name = os.path.splitext(lb)[0] + 'prof_img.png'
    profile_name = os.path.splitext(lb)[0] + '_prof.png'

    fig1 = plt.figure('fig1', figsize=(4, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('x dimension (px)')
    ax1.set_ylabel('y dimension (px)')
    plt.imshow(img, label=lb, vmin=50, vmax=1000)
    plt.plot(y_px * np.ones(len(x)), label=lb)
    plt.gca().invert_yaxis()
    ax1.legend(loc='lower right', fancybox=True, framealpha=1)
    plt.tight_layout()

    fig2 = plt.figure('fig2', figsize=(8, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('x dimension (V)')
    ax2.set_ylabel('counts')
    plt.plot(img[y_px, :], '.:', label='y V = ' + str(round(y[-1 * y_px], 4)))
    ax2.legend(loc='lower right', fancybox=True, framealpha=1)
    plt.tight_layout()
    plt.ylim((0, 1.1 * np.max(img[y_px, :])))
    plt.show()

    os.chdir(p0)
    ax1.legend(loc='lower left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
    ax2.legend(loc='lower left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
    prd.PPT_save_2d(fig1, ax1, img_name)
    prd.PPT_save_2d(fig2, ax2, profile_name)
