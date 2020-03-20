##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import scipy as sp
import scipy.constants
import scipy.optimize as opt
from scipy import ndimage


###############################################################################
# Maths defs
###############################################################################

# Functions (often used in fitting):
# Sinusoid ####################################################################
def Asinx(x, A, B, ϕ):
    y = A * np.sin(B * x + ϕ)
    return y


# Saturation curve ############################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


# Generic straight line #######################################################
def straight_line(x, m, c=0):
    y = m * x + c
    return y


# Generic 1D Gaussian peak function ###########################################
def Gaussian_1D(x, A, x_c, σ, bkg=0, N=1):
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * σ ** 2))**N) + bkg
    return G


# Generic 1D Lorentzian function ##############################################
def Lorentzian_1D(x, A, x_c, γ, bkg=0):
    L = (A * γ ** 2) / ((x - x_c)**2 + γ ** 2) + bkg
    return L


# Pseudo Voigt function as defined on wikepedia ###############################
def Voigt_1D(x, A, x_c, γ, σ, η, bkg=0):
    f_G = 2 * σ * 2 * np.sqrt(2 * np.log(2))
    f_L = 2 * γ
    f = (f_G**5 +
         2.69269 * (f_G**4) * f_L +
         2.42843 * (f_G**3) * (f_L**2) +
         4.47163 * (f_G**2) * (f_L**3) +
         0.07842 * f_L * (f_L**4) +
         f_L**5)**0.2
    V = (η * Gaussian_1D(x, A, x_c, f, bkg) +
         (1 - η) * Lorentzian_1D(x, A, x_c, f, bkg))
    return V


# Generic 2D Gaussian peak function ###########################################
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


# g2 function taken from "Berthel et al 2015" for 3 level system ##############
def g2_3_lvl(τ, δτ, a, b, c):
    g = 1 - c * np.exp(- a * np.abs(τ - δτ))
    + (c - 1) * np.exp(- b * np.abs(τ - δτ))
    return g


# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope ############################################
def g2_3_lvl_exp(τ, δτ, a, b, c, d, bkg):
    g1 = 1 - c * np.exp(- a * np.abs(τ - δτ))
    g2 = (c - 1) * np.exp(- b * np.abs(τ - δτ))
    h = (g1 + g2) * np.exp(-d * np.abs(τ - δτ)) + bkg
    return h


# Fit hologram period, Λ and rotation angle ϕ datasets from peak find #########
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


# Mode overlap for 2 fields G1 G2 in field with x & y axis ####################
def overlap(x, y, G1, G2):
    η1 = sp.trapz(sp.trapz((G1 * G2), y), x)
    η2 = sp.trapz(sp.trapz(G1, y), x) * sp.trapz(sp.trapz(G2, y), x)
    η = η1 / η2
    return η


# Pad an array A with n elements all of value a ###############################
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


# Find nearest element in array to value ######################################
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# Get Gaussian fit of histogram of data set a ################################
def Gauss_hist(a, bins=10, rng=3, res=1000):
    μ = np.mean(a)
    σ = np.sqrt(np.var(a))
    n, bins = np.histogram(a, bins)
    x = np.linspace(μ - rng * σ, μ + rng * σ, res)
    y = Gaussian_1D(x, np.max(n), μ, σ)
    return x, y


# Poissonian distribution at values of k for mean value λ #####################
def Poissonian_1D(k, λ):
    P = []

    for i0, j0 in enumerate(k):
        P.append(np.exp(-λ) * (λ**j0) / sp.math.gamma(j0 + 1))

    return P


# Dipole emission in x y field plane with strength in z direction #############
def Dipole_2D(x, y, L, λ, shift=0, I_0=1):
    # see  "Dipole antenna" wikipedia
    ρ, ϕ = cart2pol(y, x)
    S = (1 / (ρ - shift)**2) * (L**2 / λ**2) * np.sin(ϕ)**2
    return S


# Gaussian beam propagating in the +- x direction, y is the radial ############
def Gaussian_beam(z, r, w0, λ, I0=1):
    # see "Gaussian beam" wikipedia
    k = 2 * np.pi / λ
    zR = np.pi * w0**2 / λ
    w_z = w0 * np.sqrt(1 + (z / zR)**2)
    R_z = z * (1 + (zR / z)**2)
    I_rz = I0 * (w0 / w_z)**2 * np.exp((-2 * r**2) / (w_z**2))
    return I_rz, w_z[0][:]


# General monomial function (see wiki, monomial)
def Monomial(x, a, k):
    y = a * (x**k)
    return y


# dB to linear
def dB_to_lin(a):
    b = 10**(a / 10)
    return b


# linear to dB
def lin_to_dB(a):
    b = 10 * np.log10(a / 10)
    return b
