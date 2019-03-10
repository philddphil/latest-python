##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy import ndimage


###############################################################################
# Maths defs
###############################################################################
# Saturation curve ############################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


# Generic straight line #######################################################
def straight_line(x, m, c):
    y = m * x + c
    return y


# Generic 1D Gaussian function ################################################
def Gaussian_1D(x, A, x_c, σ_x, bkg=0, N=1):
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = bkg + A * np.exp(- (((x - x_c) ** 2) / (2 * σ_x ** 2))**N)
    return G


# Generic 2D Gaussian function ################################################
def Gaussian_2D(coords, A, x_c, y_c, σ_x, σ_y, θ=0, bkg=0, N=1):
    x, y = coords
    x_c = float(x_c)
    y_c = float(y_c)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    G = (bkg + A * np.exp(- (a * ((x - x_c) ** 2) +
                             2 * b * (x - x_c) * (y - y_c) +
                             c * ((y - y_c) ** 2))**N))
    return G.ravel()


# Generic 1D Lorentzian function ##############################################
def Lorentzian_1D(x, x_c, γ, A, bkg=0):
    L = (A * γ ** 2) / ((x - x_c)**2 + γ ** 2)
    return L


# Fit Λ and ϕ datasets from peak finding routine ##############################
def find_fit_peak(x, y, A, x_c):
    x_1 = np.linspace(min(x), max(x), 100)
    Peak_ind = np.unravel_index(y.argmax(), y.shape)
    initial_guess = (A, x[Peak_ind[0]], x_c, 0)

    # Fit data
    try:
        popt, pcov = opt.curve_fit(
            Gaussian_1D, x, y, p0=initial_guess)
        fit0 = Gaussian_1D(x_1, *popt)

        # After performing result, and the fitted data are saved to a temp.
        # location for labVIEW to plot
        # paths for data + fit
        p1 = (r'C:\Users\User\Documents\Phils LabVIEW\Data'
              r'\Calibration files\sweepfit.csv')
        p2 = (r'C:\Users\User\Documents\Phils LabVIEW\Data'
              r'\Calibration files\sweepdata.csv')
        # save data and fit to paths
        np.savetxt(p1, np.column_stack((x_1, fit0)), delimiter=',')
        np.savetxt(p2, np.column_stack((x, y)), delimiter=',')
        # get location of fitted peak
        Peak_ind_f = np.unravel_index(fit0.argmax(), fit0.shape)
        x_peak = x_1[Peak_ind_f[0]]

        # plt.plot(x_peak, np.max(fit0), 'x', c='xkcd:blue')
        # plt.plot(x_1, fit0, '-', c='xkcd:light blue')
        # plt.draw()
    except RuntimeError:
        print("Error - curve_fit failed")
        x_peak = 0
    return (x_peak)


# Calculate the running mean of N adjacent elements of the array x ############
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


# Gaussian blur an image n times ##############################################
def n_G_blurs(im, s=3, n=1):
    im_out = im
    for i1 in range(n):
        im_out = ndimage.filters.gaussian_filter(im_out, s)

    return im_out


# Find the RMS of array a #####################################################
def rms(a):
    b = (np.sqrt(np.mean(np.square(a))))
    return b


# Polar to cartesian coords ###################################################
def pol2cart(ρ, ϕ):
    x = ρ * np.cos(ϕ)
    y = ρ * np.sin(ϕ)
    return(x, y)


# Cartesian to polar coords ###################################################
def cart2pol(x, y):
    ρ = np.sqrt(x**2 + y**2)
    ϕ = np.arctan2(y, x)
    return(ρ, ϕ)


# Return the element location of the max of array a ###########################
def max_i_2d(a):
    b = np.unravel_index(a.argmax(), a.shape)
    return(b)


# Circle at location x, y radius r ############################################
def circle(r, x, y):
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, 100)

    # compute xc and yc
    xc = r * np.cos(theta) + x
    yc = r * np.sin(theta) + y
    return (xc, yc)


# Mode overlap for 2 fields G1 G2 in field with x & y axis
def overlap(x, y, G1, G2):
    η1 = sp.trapz(sp.trapz((G1 * G2), y), x)
    η2 = sp.trapz(sp.trapz(G1, y), x) * sp.trapz(sp.trapz(G2, y), x)
    η = η1 / η2
    return η


# Pad an array A with n elements all of value a
def Pad_A_elements(A, n, a=0):
    Ax, Ay = np.shape(A)
    P = a * np.ones(((2 * n + 1) * (Ax), (2 * n + 1) * (Ay)))
    Px, Py = np.shape(P)
    for i1 in range(Px):
        for i2 in range(Py):
            if ((i1 - n) % (2 * n + 1) == 0 and
                    (i2 - n) % (2 * n + 1) == 0):
                P[i1, i2] = A[(i1 - n) // (2 * n + 1),
                              (i2 - n) // (2 * n + 1)]
    return P


# Find nearest element in array to value
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# Get Gaussian fir of histogram of data set a
def Gauss_hist(a, bins=10, rng=3, res=1000):
    μ = np.mean(a)
    σ = np.sqrt(np.var(a))
    n, bins = np.histogram(a, bins)
    x = np.linspace(μ - rng * σ, μ + rng * σ, res)
    y = Gaussian_1D(x, np.max(n), μ, σ)
    return x, y

