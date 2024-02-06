# %% 0. imports
import os
import numpy as np
import scipy as sp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
# %% 1. defs
def cart2pol(x, y):
    """convert x y coords to cylindrical polar coords

    Args:
        x (array): x coords
        y (array): y coords

    Returns:
        rho, phi (tuple (array, array)): rho phi coords
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """convert cylindrical polar values to cartesian

    Args:
        rho (array): radial displacement in cylindrical coords
        phi (array): angle in cylindrical coords

    Returns:
        x, y (tuple, (array, array)): x y coords
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def rbytheta(x, y, x_shift, y_shift, theta):
    """rotate a set of x & y coords by angle theta 

    Args:
        x (array): x coords 
        y (array): y coords
        theta (float): angular rotation (theta = 0 is positive y direction, positive = c.w.)

    Returns:
        (Rx, Ry) (tuple(array,array)): rotated x and y coords 
    """
    rho, phi = cart2pol(x - x_shift, y - y_shift)
    Rx, Ry = pol2cart(rho, phi - theta)
    return (Rx + x_shift, Ry + y_shift)


def set_figure(name: str = 'figure',
               xaxis='x axis',
               yaxis='y axis',
               size: float = 4,
               ):
    """
    Set a figure up
    Parameters
    ----------
    name : str
        name of figure
    xaxis : str
       x axis label
    yaxis : str
       y axis label
    size : float

    Returns
    -------
    ax1 : axes._subplots.AxesSubplot
    fig1 : figure.Figure
    """
    ggplot_sansserif()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor('xkcd:dark grey')
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1


def ggplot_sansserif():
    """
    Set some parameters for ggplot
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 8
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['axes.labelcolor'] = 'xkcd:black'
    plt.rcParams['xtick.color'] = 'xkcd:black'
    plt.rcParams['ytick.color'] = 'xkcd:black'
    plt.rcParams['axes.edgecolor'] = 'xkcd:black'
    plt.rcParams['savefig.edgecolor'] = 'xkcd:black'
    plt.rcParams['axes.facecolor'] = 'xkcd:dark gray'
    plt.rcParams['savefig.facecolor'] = 'xkcd:dark gray'
    plt.rcParams['grid.color'] = 'xkcd:dark gray'
    plt.rcParams['grid.linestyle'] = 'none'
    plt.rcParams['axes.titlepad'] = 6


def Gaussian_beam(z, r, z0, w0, λ, I0=0):
    """ see "Gaussian beam" wikipedia

    Args:
        z (array): z coords
        r (array): r coords
        z0 (float): beam waist z
        w0 (float): beam waist
        I0 (int, optional): Intensity. Defaults to 0, corresponding to unity power.
    """

    k = 2 * np.pi / λ
    zR = np.pi * w0**2 / λ
    w_z = w0 * np.sqrt(1 + ((z - z0) / zR)**2)
    R_z = (z - z0) * (1 + (zR / (z - z0))**2)
    Guoy_z = np.arctan(z/zR)
    if I0 == 0:
        I0 = 2/(np.pi * w0 ** 2)

    E0 = np.sqrt(I0)

    I_rz = (I0 
            * (w0 / w_z)**2 
            * np.exp((-2 * r**2) / (w_z**2)))
    E_rz = np.real((E0 
            * (w0 / w_z) 
            * np.exp((-2 * r**2) / (w_z**2)) 
            * np.exp(-1j * (k*z + (k * r**2 / (2*R_z)) - Guoy_z))))
    
    return I_rz, E_rz

def extents(f):
    """ calculates and returns suitable values for imshow's extent from axis array

    Args:
        f (array): axis array

    Returns:
        extents : tuple for extent in imshow
    """
    delta = f[1] - f[0]
    return f[0] - delta / 2, f[-1] + delta / 2


# %% 2. tests
L = 30e-6
R = 20e-6
d = 10e-6
wavelength_m = 780e-9

# simulation region and resolution
r_range = 0.8*R
z_range = 1.5*L
r_points = 1000
z_points = 2000

# initially work in mirror coords - (0,0) bottom centre of m0
# this is the tilt of mirror 1 w.r.t mirror 0
theta1 = 0.1
# theta1 = np.pi/50 # <----- Change this one

# coordinates of mirror 1 centre of RoC
r1_c = 5e-6 # <----- Change this one
z1_c = L-R

# coordinates of mirror 0 centre of RoC (r0_c must = 0 for now)
r0_c = 0
z0_c = R
r1_c_act = L*np.tan(theta1) + r1_c
print('offset of m1 from 0 = ', np.round(1e6*r1_c_act,2), ' μm')

# height/depth of feature
h = R - np.sqrt(R**2 - ((d**2)/4))

# establish surfaces
rs = np.linspace(-r_range,
                 r_range,
                 r_points)
r_res = rs[1]-rs[0]
rs_pos = [i0 for i0 in rs if i0 > 0]
zs = np.linspace(-z_range/2 + L/2,
                 z_range/2 + L/2,
                 z_points)
z_res = zs[1]-zs[0]

# angular tilt associated with lateral offset
theta2 = np.arctan((r1_c-r0_c)/(z0_c-z1_c))

# Surface of mirror 1
C1_z_semi = np.sqrt(np.abs(R**2 - (rs-r1_c)**2)) + z1_c
# Flat surface at height L - h
S1 = (L - h)*np.ones_like(rs)
# x indices where semi-circle crosses surface
idx1 = np.argwhere(np.diff(np.sign(C1_z_semi - S1))).flatten()
C1_r = rs[idx1[0]:idx1[-1]]
C1_z = C1_z_semi[idx1[0]:idx1[-1]]
C1R_r, C1R_z = rbytheta(C1_r,
                      C1_z,
                      r1_c,
                      z1_c,
                      theta1)

print(1e6*(np.abs(C1_r[0]-C1_r[-1])))
C2_dr = C1_r[0]-C1_r[-1]
C2_dz = C1_z[0]-C1_z[-1]
C2_chord = np.sqrt(C2_dr**2 + C2_dz**2)

# % 4. Gaussian beam calcs
# calc 'normal' TEM00 mode parameters - NOTE symmetric cavity assumed (RoC of faces are the same)
# Gaussian beam parameters (see wikipedia)
zR = np.sqrt((R/(L/2)-1))*(L/2)
n = np.floor(L/wavelength_m)
l = L/n
w0 = np.sqrt((zR * l) / (np.pi))
wm = w0 * np.sqrt(1+(L/(2*zR))**2)
kx = (zs-L/2)*np.tan(theta2)+r1_c/2
mode_c_r = (r1_c - r0_c)/2
mode_c_z = L/2

coords = np.meshgrid(zs, rs)
# rotate each coordinate grid for irradiance calculations
Rcoords = rbytheta(coords[0]-z1_c, coords[1]-r1_c, 0, 0, theta1)
# calculate the irradiance & field in each rotated coord system
GI, GE = Gaussian_beam(Rcoords[0], Rcoords[1], 0, w0, l, 1)

# Plots
ax1, fig1 = set_figure('mirror arrangement',
                       'r / μm',
                       'z / μm')
img = ax1.imshow(np.transpose(GI/np.max(GI)),
                 extent=extents(1e6*rs) + extents(1e6*zs),
                 #    norm=LogNorm(vmin=1e-6, vmax=1e6),
                 cmap= 'Pastel2',
                 origin='lower',
                 )
ax1.plot(1e6*C1_r,
         1e6*C1_z,
         '-',
         color='xkcd:blue',
         )

ax1.plot(1e6*C1R_r, 1e6*C1R_z,
         ':',
         color='xkcd:red',
         label='m1')
ax1.plot(1e6*rs,1e6*mode_c_z*np.ones_like(rs),
         '-',
         color='xkcd:black')
ax1.set_ylim(bottom=0)
ax1.set_aspect('equal')