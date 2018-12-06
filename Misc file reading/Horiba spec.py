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
# Specify results directory and change working directory to this location
p0 = (r"D:\Experimental Data\Spectrometer (Ganni F5 L10)\Spec data 20181113")
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
    λ, cts = prd.load_Horiba(val0)
    lb = os.path.basename(val0)
    plot_name = os.path.splitext(lb)[0] + '_img.png'

    # append each data set to list for subsequent plotting
    λs.append(λ)
    ctss.append(cts)

    # plot each data set and save (close pop-up to save each time)
    prd.ggplot()
    fig1 = plt.figure('fig1', figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('Wavelength (λ) / nm')
    ax1.set_ylabel('Counts')
    ax1.plot(λ, cts, '.', alpha=0.2, label=lb)
    ax1.plot(λ, prd.n_G_blurs(cts, 5), label='smoothed')
    plt.ylim((0, 1.1 * np.max(cts)))
    plt.title('spectrum')
    plt.tight_layout()
    ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.show()
    ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
    prd.PPT_save_2d(fig1, ax1, plot_name)

# Do any dataset manipulation required
diff = ctss[0] - ctss[1]

# Final plots
fig1 = plt.figure('fig1', figsize=(6, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Counts')
ax1.plot(λ, ctss[0], '.', color=cs['gglred'], alpha=0.2, label='bkg')
ax1.plot(λ, prd.n_G_blurs(ctss[0], 5), color=cs['ggred'], label='smoothed')
ax1.plot(λ, ctss[1], '.', color=cs['gglblue'], alpha=0.2, label='peak')
ax1.plot(λ, prd.n_G_blurs(ctss[1], 5), color=cs['ggblue'], label='smoothed')
ax1.plot(λ, diff, '.', color=cs['gglpurple'], alpha=0.2, label='peak')
ax1.plot(λ, prd.n_G_blurs(diff, 5), color=cs['ggpurple'], label='smoothed')
ax1.legend(loc='upper right', fancybox=True, framealpha=1)
plt.title('spectrum')
plt.tight_layout()
plt.show()
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig1, ax1, 'Spectrum.png')
