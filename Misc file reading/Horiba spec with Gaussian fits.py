##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
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
a = 1

λs, ctss, lbs = prd_file_import.load_spec_dir(p0)
x = λs[a]
y = ctss[a]

# Use gin to get first approximation for peak location
pts = prd_plots.gin(λs[0], ctss[0], 0)
λ_pk, idx_pk = prd_maths.find_nearest(x, pts[0, 0])
# Restrict data set to roi of interest
x_roi = x[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
y_roi = y[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
print(np.shape(x_roi))
print(np.shape(y_roi))
# Extract first guess values for fitting
mean = λ_pk
sigma = 0.1
bkg = np.mean(y_roi)
print([1, mean, sigma, bkg])
# Set up higher resolution x axis for fit
x_fit = np.linspace(min(x_roi), max(x_roi), 1000)
# Perform fit
popt, pcov = curve_fit(prd_maths.Gaussian_1D,
                       x_roi, y_roi, p0=[1, mean, sigma, bkg])

# Plot 
prd_plots.ggplot()
size = 4
colors = plt.cm.viridis(np.linspace(0, 1, len(λs)))

fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Counts')

for i1, val in enumerate(λs):
    x = λs[i1]
    y = ctss[i1]
    x_roi = x[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
    y_roi = y[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]

    n = len(x_roi)
    mean = λ_pk
    sigma = 0.1
    bkg = np.mean(y_roi)

    x_fit = np.linspace(min(x_roi), max(x_roi), 1000)

    popt, pcov = curve_fit(prd_maths.Gaussian_1D,
                           x_roi, y_roi, p0=[popt])

    # Final plots
    prd_plots.ggplot()

    colors = plt.cm.viridis(np.linspace(0, 1, len(λs)))
    ax1.plot(x, y, '--', alpha=0.5,
             color=colors[i1],
             label='',
             lw=0)

    plt.plot(x_roi, y_roi, '.',
             c=colors[i1],
             alpha=0.3)
    plt.plot(x_fit, prd_maths.Gaussian_1D(
        x_fit, *popt),
        label='fit',
        c=colors[i1],
        lw=0.5)

    plt.xlim((x[idx_pk - int(0.3 * roi)], x[idx_pk + int(0.3 * roi)]))
ax1.set_title('spectra with fits')
fig1.tight_layout()
plt.show()

ax1.figure.savefig('spectra with fits dark' + '.png')
prd_plots.PPT_save_2d(fig1, ax1, 'spectra with fits')
