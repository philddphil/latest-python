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
# p0 = (r"D:\Experimental Data\F5 L10 Spectrometer\Spec data 20190403")
p0 = (r"D:\Experimental Data\Internet Thorlabs optics data")
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*DMLP550 T.txt')
datafiles.sort(key=os.path.getmtime)

# Initialise lists of datasets
λs = []
ctss = []
count = 0

for i0, val0 in enumerate(datafiles[0:]):
    # load each spec file, generate λ array cts array, label and plot name
    cts = []
    λ, cts = prd_file_import.load_spec(val0)
    lb = os.path.basename(val0)
    plot_name = '925 to 945 ' + os.path.splitext(lb)[0]

    for i1, val1 in enumerate(cts[0, :]):
        lb = str(i1) + ' ' + os.path.basename(val0)
        print(count, lb)
        cts_crr = prd_data_proc.cos_ray_rem(cts[:, i1], 50)
        ctss.append(list(cts_crr))
        λs.append(λ)
        # append each data set to list for subsequent plotting
        y0 = cts_crr
        # plot each data set and save (close pop-up to save each time)
        prd_plots.ggplot()
        fig1 = plt.figure('fig1', figsize=(3 * np.sqrt(2), 3))
        ax1 = fig1.add_subplot(1, 1, 1)
        fig1.patch.set_facecolor(cs['mnk_dgrey'])
        ax1.set_xlabel('Wavelength (λ) / nm')
        ax1.set_ylabel('Counts')
        ax1.plot(λ, y0, '.', alpha=0.4, color=cs['gglred'], label=lb)
        ax1.plot(λ, y0, alpha=1, color=cs['ggdred'], lw=0.5, label='')
        # plt.ylim(1.1 * np.min(y0), 1.1 * np.max(y0))
        plt.xlim((530, 800))
        plt.title('spectrum')
        plt.tight_layout()
        ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
        plt.show()
        ax1.figure.savefig(plot_name + 'dark.png')
        ax1.legend(loc='upper left', fancybox=True,
                   facecolor=(1.0, 1.0, 1.0, 0.0))
        prd_plots.PPT_save_2d(fig1, ax1, plot_name)
        count += 1

# Do any dataset manipulation required
# diff = ctss[0] - ctss[1]

# Final plots
a = 0
b = 1
c = 5
prd_plots.ggplot()
fig1 = plt.figure('fig1', figsize=(6, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('% Transmission')
ax1.plot(λs[a], ctss[a], '.', markersize=1,
         alpha=0.5, color=cs['gglred'], label='NPBS NIR P')
ax1.plot(λs[a], ctss[a], alpha=0.8,
         color=cs['ggdred'], lw=0.5, label='')
ax1.plot(λs[b], ctss[b], '.', markersize=1,
         alpha=0.5, color=cs['gglblue'], label='NPBS NIR S')
ax1.plot(λs[b], ctss[b], alpha=0.8,
         color=cs['ggdblue'], lw=0.5, label='')
# ax1.plot(λs[c], ctss[c], '.', markersize=1,
#          alpha=0.5, color=cs['ggpurple'], label='QD3')
# ax1.plot(λs[c], ctss[c], alpha=0.8,
#          color=cs['gglpurple'], lw=0.5, label='')
# ax1.plot(λs[7], 10 * ctss[7], color=cs['ggdyellow'],
#          alpha=0.5, label='filter')
# ax1.plot(λs[7], 10 * ctss[7], '.', color=cs['ggyellow'], label='')
ax1.legend(loc='upper right', fancybox=True, framealpha=1)
plt.title('spectrum')
plt.tight_layout()
plt.xlim((530, 800))
plt.show()
ax1.figure.savefig('Spectrum_dark' + '.png')
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, 'Spectrum')
