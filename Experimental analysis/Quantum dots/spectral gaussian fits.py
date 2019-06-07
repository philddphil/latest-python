##############################################################################
# Import some libraries
##############################################################################
import re
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
xs0 = λs[a]
ys0 = ctss[a]

# Use gin to get first approximation for peak location
pts = prd_plots.gin(λs[0], ctss[0], 0)

for i1, val1 in enumerate(pts):
    pk_lb = 'peak ' + str(i1)
    λ_pk, idx_pk = prd_maths.find_nearest(xs0, pts[i1, 0])

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
    As_all.append(As)
    μs_all.append(μs)
    σs_all.append(σs)
    Ps_all.append(Ps)
