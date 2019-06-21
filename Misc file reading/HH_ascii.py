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
import prd_data_proc
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 HydraHarp\HH data 20190614")
os.chdir(p0)
datafiles = glob.glob(p0 + r'\*.dat')
for i0, val0 in enumerate(datafiles):
    lb = os.path.splitext(os.path.basename(val0))[0]
    print(lb)
    τs0, τs1, cts0, cts1 = prd_file_import.load_HH(datafiles[i0])
    τs0_ss, cts0_ss = prd_data_proc.HH_data_subset(τs0, cts0, 48, 200)
    size = 4
    prd_plots.ggplot()
    fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('τ (ns)')
    ax1.set_ylabel('Counts')

    ax1.bar(τs0_ss, cts0_ss, width=τs0[1], edgecolor=cs['mnk_dgrey'])
    plt.show()
    plot_name = lb
    prd_plots.PPT_save_2d(fig1, ax1, plot_name)
