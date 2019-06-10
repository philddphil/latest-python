##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# Specify results directory and change working directory to this location
p0 = (r"D:\Experimental Data\F5 L10 Spectrometer\Spec data 20190516")
# p0 = (r"D:\Experimental Data\Internet Thorlabs optics data"))
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
roi = 80

λs, ctss, lbs = prd_file_import.load_spec_dir(p0)
xs0 = λs[1]
ys0 = ctss[1]

# Use gin to get first approximation for peak location
pts = prd_plots.gin(λs[0], ctss[0], 0)

pk_λs = []
pk_idxs = []

# Loop over data in directory and perform fits on each spec, for each peak
for i0, val0 in enumerate(pts):
    pk_λ = str(int(np.round(pts[i0][0])))
    pk_lb = 'peak ' + str(i0) + ' (' + pk_λ + ' nm)'
    λ_pk, idx_pk = prd_maths.find_nearest(xs0, pts[i0, 0])
    pk_λs.append(λ_pk)
    pk_idxs.append(idx_pk)
    # Restrict data set to roi of interest
    x_roi = xs0[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
    y_roi = ys0[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
    # Extract first guess values for fitting
    μ = λ_pk
    σ = 0.1
    bkg = np.mean(y_roi)
    # Set up higher resolution x axis for fit
    x_fit = np.linspace(min(x_roi), max(x_roi), 1000)
    # Perform fit
    popt, pcov = curve_fit(prd_maths.Gaussian_1D,
                           x_roi, y_roi, p0=[1, μ, σ, bkg])

    As, μs, σs, Ps = prd_data_proc.spec_seq_Gauss_fit(p0,
                                                      popt,
                                                      idx_pk,
                                                      roi,
                                                      pk_lb)

    data_name = pk_lb + '.dat'
    data = np.column_stack((Ps, As, μs, σs))
    header = "Powers, Gaussian Amplitudes, Gaussian centres, Gaussian widths"
    np.savetxt(data_name, data, header=header)

prd_plots.ggplot()
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ - nm)')
ax1.set_ylabel('Counts')
ax1.set_title('Labelled spectrum with fits')
ax1.plot(xs0, ys0, '.', markersize=2,
         alpha=0.5, color=cs['gglred'], label='')
pk_xs = [xs0[i] for i in pk_idxs]
pk_ys = [ys0[i] for i in pk_idxs]
for i0, val0 in enumerate(pk_xs):
    ax1.plot(pk_xs[i0], pk_ys[i0], 'o', mfc=cs['ggblue'], label='peak ' + str(i0))

fig1.tight_layout()
ax1.legend(loc='upper right', fancybox=True, framealpha=1)
plt.show()

ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, 'peak labels.png')