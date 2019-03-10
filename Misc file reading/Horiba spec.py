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
import prd_plots
import prd_file_import
import prd_data_proc
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# Specify results directory and change working directory to this location
p0 = (r"D:\Experimental Data\Spectrometer (Ganni F5 L10)\Spec data 20190306")
os.chdir(p0)

# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)

# Initialise lists of datasets
λs = []
ctss = []

for i0, val0 in enumerate(datafiles[0:]):
    # load each spec file, generate λ array cts array, label and plot name
    print(i0, val0)
    λ, cts = prd_file_import.load_Horiba(val0)
    lb = os.path.basename(val0)
    plot_name = os.path.splitext(lb)[0]
    cts_crr = prd_data_proc.cos_ray_rem(cts, 30)
    # append each data set to list for subsequent plotting
    λs.append(λ)
    ctss.append(cts_crr)
    y0 = cts_crr
    # # plot each data set and save (close pop-up to save each time)
    prd_plots.ggplot()
    fig1 = plt.figure('fig1', figsize=(3 * np.sqrt(2), 3))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('Wavelength (λ) / nm')
    ax1.set_ylabel('Counts')
    ax1.plot(y0, '.', alpha=0.4, color=cs['gglred'], label=lb)
    ax1.plot(y0, alpha=1, color=cs['ggdred'], lw=0.5, label='')
    plt.ylim(1.1 * np.min(y0), 1.1 * np.max(y0))
    # plt.xlim((890, 910))
    plt.title('spectrum')
    plt.tight_layout()
    ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.show()
    ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
    ax1.figure.savefig(plot_name + '.png')
    prd_plots.PPT_save_2d(fig1, ax1, plot_name)

# Do any dataset manipulation required
# diff = ctss[0] - ctss[1]

# Final plots
# prd_plots.ggplot()
# fig1 = plt.figure('fig1', figsize=(6, 4))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('Wavelength (λ) / nm')
# ax1.set_ylabel('Counts')
# ax1.plot(λs[2], ctss[2], color=cs['gglred'], alpha=0.2, label='QD3')
# ax1.plot(λs[2], ctss[2], '.', color=cs['ggred'], label='')
# ax1.plot(λs[3], ctss[3], color=cs['gglblue'], alpha=0.2, label='QD2')
# ax1.plot(λs[3], ctss[3], '.', color=cs['ggblue'], label='')
# ax1.plot(λs[4], ctss[4], color=cs['ggpurple'], alpha=0.5, label='QD1')
# ax1.plot(λs[4], ctss[4], '.', color=cs['ggpurple'], label='')
# ax1.plot(λs[7], 10 * ctss[7], color=cs['ggdyellow'],
#          alpha=0.5, label='filter')
# ax1.plot(λs[7], 10 * ctss[7], '.', color=cs['ggyellow'], label='')
# ax1.legend(loc='upper right', fancybox=True, framealpha=1)
# plt.title('spectrum')
# plt.tight_layout()
# plt.xlim((890, 910))
# plt.show()
# ax1.figure.savefig('Spectrum' + '.png')
# ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
# prd_plots.PPT_save_2d(fig1, ax1, 'Spectrum')
