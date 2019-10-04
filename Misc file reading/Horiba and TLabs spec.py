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
# p0 = (r"D:\Experimental Data\F5 L10 Spectrometer\Spec data 20190815")
p0 = (r"D:\Experimental Data\Internet data\Thorlabs optics\FBH850-10")
p0 = (r"D:\Experimental Data\Internet data\Thorlabs optics\FEL0800")
p0 = (r"D:\Experimental Data\Internet data\Thorlabs optics\FGL780")
p0 = (r"D:\Experimental Data\Internet data\Thorlabs optics\FGL830")
p0 = (r"D:\Experimental Data\Internet data\Thorlabs optics\DMLP550\DMSP950")
p0 = (r"D:\Experimental Data\Internet data\Solar spec")
# p0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Conferences\SPW 2019\Figures")

os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)

# Initialise lists of datasets
λs = []
ctss = []
lbs = []
count = 0

for i0, val0 in enumerate(datafiles[0:]):
    # load each spec file,
    # generate λ array,
    # generate cts array,
    # generate label and plot name
    cts = []
    λ, cts = prd_file_import.load_spec(val0)
    lb = os.path.basename(val0)
    plot_name = '' + os.path.splitext(lb)[0]

    for i1, val1 in enumerate(cts[0, :]):
        lb = str(i1) + ' ' + os.path.basename(val0)
        lbs.append(os.path.splitext(lb)[0])
        print(count, os.path.splitext(lb)[0])

        cts_crr = prd_data_proc.cos_ray_rem(cts[:, i1], 50)
        ctss.append(list(cts_crr))
        λs.append(λ)
        # append each data set to list for subsequent plotting
        y0 = cts_crr / np.max(cts_crr)

        # plot each data set and save (close pop-up to save each time)

        prd_plots.ggplot()
        fig1 = plt.figure('fig1', figsize=(3 * np.sqrt(2), 3))
        ax1 = fig1.add_subplot(1, 1, 1)
        fig1.patch.set_facecolor(cs['mnk_dgrey'])
        ax1.set_xlabel('Wavelength (λ) / nm')
        ax1.set_ylabel('Transmission')
        ax1.plot(λ, y0, '.', alpha=0.4, color=cs['gglred'],
                 label=os.path.splitext(lb)[0])
        ax1.plot(λ, y0, alpha=1, color=cs['ggdred'], lw=0.5, label='')
        ax1.plot([852, 852], [1.1, 0],
                 color=cs['ggblue'], label='Q channel')
        ax1.plot([1064, 1064], [1.1, 0],
                 color=cs['ggyellow'], label='Comm channel')
        ax1.plot([633, 633], [1.1, 0],
                 color='xkcd:bright red', label='633 nm')
        ax1.plot([532, 532], [1.1, 0],
                 color='xkcd:bright green', label='532 nm')
        ax1.fill_between([700, 1000], [0.5, 0.5], [0, 0], alpha=0.5,
                         color=cs['mnk_pink'], label='Ti:Sapph')
        # plt.ylim(1.1 * np.min(y0), 1.1 * np.max(y0))
        plt.xlim((200, 2500))
        plt.title('spectra')
        plt.tight_layout()
        ax1.legend(loc='upper right', fancybox=True, framealpha=0.5)
        plt.show()
        ax1.figure.savefig(plot_name + 'dark.png')
        ax1.legend(loc='upper right', fancybox=True,
                   facecolor=(1.0, 1.0, 1.0, 0.0))
        prd_plots.PPT_save_2d(fig1, ax1, plot_name)

        # increment counter
        count += 1

# Do any dataset manipulation required
# diff = ctss[0] - ctss[1]

# Final plots
# a = 0
# b = 1
# c = 2

# x0 = λs[0]
# x1 = λs[1]
# x2 = λs[2]

# y0 = (ctss[0] - np.min(ctss[0])) / np.max(ctss[0])
# y1 = (ctss[1] - np.min(ctss[1])) / np.max(ctss[1])
# y2 = (ctss[2] - np.min(ctss[2])) / np.max(ctss[2])

# prd_plots.ggplot()
# # fig1 = plt.figure('fig1', figsize=(5 * np.sqrt(2), 5))
# fig1 = plt.figure('fig1', figsize=(8, 4))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('Wavelength (λ) / nm')
# ax1.set_ylabel('Arbitrary Units')

# # plt.imshow(ctss, cmap='magma',
# #            extent=prd_plots.extents(x) +
# #            prd_plots.extents(y),
# #            aspect='auto')
# # for i0 in np.arange(45):
# #     ax1.plot(λs[i0] + 0 * i0, ctss[i0], '-o',
# #              markersize=1, alpha=0.5, label=i0)
# ax1.plot(x0, y0, '.', markersize=1,
#          alpha=0.5, color=cs['gglred'], label='')
# ax1.plot(x0, y0, alpha=0.8,
#          color=cs['ggdred'], lw=1, label='QD')
# ax1.plot(x1, y1, '.', markersize=1,
#          alpha=0.5, color=cs['gglblue'], label='')
# ax1.plot(x1, y1, alpha=0.8,
#          color=cs['ggdblue'], lw=1, label='NV')
# ax1.plot(x2, y2, '.', markersize=1,
#          alpha=0.5, color=cs['ggpurple'], label='')
# ax1.plot(x2, y2, alpha=0.8,
#          color=cs['gglpurple'], lw=1, label='hBN')
# # ax1.plot(λs[7], 10 * ctss[7], color=cs['ggdyellow'],
# # #          alpha=0.5, label='filter')
# # # ax1.plot(λs[7], 10 * ctss[7], '.', color=cs['ggyellow'], label='')
# ax1.legend(loc='upper right', fancybox=True, framealpha=1)
# # plt.title('spectrum')
# plt.tight_layout()
# # plt.xlim((500, 1000))
# plt.show()
# ax1.legend(loc=, fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
# ax1.figure.savefig('Spectrum_dark' + '.png')
# prd_plots.PPT_save_2d(fig1, ax1, 'Spectrum')
