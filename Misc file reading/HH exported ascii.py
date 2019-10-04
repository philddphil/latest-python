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
sys.path.insert(0, r"D:\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = r"D:\Experimental Data\F5 L10 HydraHarp\HH data 20190912\\"

##############################################################################
# Plot some figures
##############################################################################
plot_path = r"D:\Python\Plots\\"

###### xy plot ###############################################################
datafiles = glob.glob(p0 + r'\*.dat')
for i0, val0 in enumerate(datafiles[0:]):
    lb = os.path.basename(val0)
    lb = os.path.splitext(lb)[0]
    ts, cts = prd_file_import.load_HH(val0)

    prd_plots.ggplot()
    size = 4
    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(111)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Δt (ns)')
    ax2.set_ylabel('freq #')
    plt.plot(ts, cts, lw=0.5,
             alpha=0.4, color=cs['ggdred'], label=lb)
    plt.plot(ts, cts, '.', lw=0.5,
             alpha=0.8, color=cs['gglred'], markersize=0.5)
    plt.xlim((90, 150))
    plt.ylim((0, max(cts)))
    ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.show()
    plot_file_name = p0 + lb + '.png'    
    ax2.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
    prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)

