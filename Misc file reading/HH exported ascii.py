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
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
import prd_maths
import prd_file_import
import prd_plots
np.set_printoptions(suppress=True)
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = r"C:\local files\Experimental Data\F5 L10 HydraHarp\HH data 20200113\\"
# p0 = r"C:\Users\pd10\OneDrive - National Physical Laboratory\Conferences\SPW 2019\Data\HBTs"
os.chdir(p0)
##############################################################################
# Plot some figures
##############################################################################


###### xy plot ###############################################################
datafiles = glob.glob(p0 + r'\*.dat')
for i0, val0 in enumerate(datafiles[0:]):
    lb = os.path.basename(val0)
    lb = os.path.splitext(lb)[0]
    ts, cts = prd_file_import.load_HH(val0)
    # norm = np.mean(cts[0:50])
    prd_plots.ggplot()
    size = 2.5
    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(111)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Δt (ns)')
    ax2.set_ylabel('g$^2$(τ)')
    ax2.set_ylabel('counts')
    t_shift=93
    plt.plot(ts-t_shift, cts, lw=0.5,
             alpha=0.4, color=cs['ggdred'], label=lb)
    plt.plot(ts-t_shift, cts, 'o', lw=0.5,
             alpha=1, color=cs['gglred'], markersize=2)
    plt.xlim((-t_shift, 500))
    plt.ylim((0, np.max(cts)))
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.show()
    plot_file_name = p0 + lb + '.svg'
    # ax2.legend(loc='upper right', fancybox=True,
    #            facecolor=(1.0, 1.0, 1.0, 0.0))
    ax2.figure.savefig(plot_file_name)
