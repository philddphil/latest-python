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
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
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
p0 = (r"C:\local files\Experimental Data\F5 L10 Spectrometer\Spec data 20190516")
# p0 = (r"D:\Experimental Data\Internet Thorlabs optics data"))
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafile = p0 + r'\30mA.txt'
roi = 80
λ, cts = prd_file_import.load_spec(datafile)
print('resolution = ', λ[1] - λ[0], ' nm')
cts = cts[:, 0]
# Use gin to get first approximation for peak location
pts = prd_plots.gin(λ, cts, 0, 'click once and tap enter to fit peak')
λ_pk, idx_pk = prd_maths.find_nearest(λ, pts[0, 0])
print(λ_pk, idx_pk)
# Restrict data set to roi of interest
x_roi = λ[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
y_roi = cts[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
# Extract first guess values for fitting
mean = λ_pk
sigma = 0.1
bkg = np.mean(y_roi)
# Set up higher resolution x axis for fit
x_fit = np.linspace(min(x_roi), max(x_roi), 1000)
# Perform fit
popt_G, pcov = curve_fit(prd_maths.Gaussian_1D,
                         x_roi, y_roi, p0=[1, mean, sigma, bkg])

popt_L, pcov = curve_fit(prd_maths.Lorentzian_1D,
                         x_roi, y_roi, p0=[mean, sigma, 1, bkg])
print('Gaussian FWHM = ', np.abs(np.round(2.354 * popt_G[2], 3)), 'nm')
print('Lorentzian FWHM = ', np.abs(np.round(2 * popt_L[1], 3)), 'nm')

# Plots
prd_plots.ggplot()
size = 4

fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Counts')

fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Wavelength (λ) / nm')
ax2.set_ylabel('Residuals')


x = λ
y = cts
# Final plots
prd_plots.ggplot()

ax1.plot(x_roi, y_roi, '.',
         label='data',
         color=cs['ggred'],
         alpha=1)

ax1.plot(x_fit, prd_maths.Gaussian_1D(
    x_fit, *popt_G),
    color=cs['ggblue'],
    label='Gaussian fit',
    lw=0.5)
ax1.plot(x_fit, prd_maths.Lorentzian_1D(
    x_fit, *popt_L),
    color=cs['ggpurple'],
    label='Lorentzian fit',
    lw=0.5)

ax2.plot(x, y - prd_maths.Gaussian_1D(
    x, *popt_G), '.-',
    color=cs['ggblue'],
    label='Gaussian residuals',
    lw=0.5)

ax2.plot(x, y - prd_maths.Lorentzian_1D(
    x, *popt_L), '.-',
    color=cs['ggpurple'],
    label='Lorentzian residuals',
    lw=0.5)

ax1.set_xlim((x[idx_pk - int(0.3 * roi)], x[idx_pk + int(0.3 * roi)]))
ax2.set_xlim((x[idx_pk - int(0.3 * roi)], x[idx_pk + int(0.3 * roi)]))
ax2.set_ylim((-1000, 1000))
ax1.legend(loc='upper right', fancybox=True, framealpha=0.5)
ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
ax1.set_title('spectrum with fits')
ax2.set_title('residuals')
fig1.tight_layout()
fig2.tight_layout()
plt.show()
ax1.figure.savefig('spectrum with fits dark' + '.png')
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, 'spectrum with fits 30mA')
prd_plots.PPT_save_2d(fig2, ax2, 'residuals 30mA')
