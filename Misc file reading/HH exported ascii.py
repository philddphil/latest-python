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
# Specify directory with the .dat files for plotting
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20200127\HH")

# change current working directory to path above
os.chdir(p0)

# Generate list of files with .dat extension in directory
datafiles = glob.glob(p0 + r'\*.dat')

##############################################################################
# Directory loop so everythin is done in a for loop
##############################################################################
for i0, v0 in enumerate(datafiles[0:]):

    # create label from filename
    lb = os.path.basename(v0)
    lb = os.path.splitext(lb)[0]

    # import x/y data from filename (v0)
    ts, cts = prd_file_import.load_HH(v0)
    
    # set plot parameters
    prd_plots.ggplot()
    size = 2.5

    # begin populating figure
    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(111)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Δt (ns)')
    ax2.set_ylabel('counts')
    t_shift = 93
    plt.plot(ts - t_shift, cts, lw=0.5,
             alpha=0.4, color=cs['ggdred'], label=lb)
    plt.plot(ts - t_shift, cts, 'o', lw=0.5,
             alpha=1, color=cs['gglred'], markersize=2)
    plt.xlim((-t_shift, 500))
    plt.ylim((0, np.max(cts)))
    plt.tight_layout()

    # show figure (pop up)
    plt.show()

    # generate filename
    plot_file_name = lb

    # save figure
    prd_plots.PPT_save_2d(fig2, ax2, lb)
