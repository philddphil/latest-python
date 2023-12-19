# purpose of code
# calculate optimal alignment of cavities given
# angular mis-alignment
# %% imports
import os
import numpy as np
import scipy as sp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

# %% defs
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


def PPT_save_plot(fig, ax, name, dpi=600):
    """ saves a plot as 'name'.png (unless .svg is specified in the name). 
    Iterates name (name_0, name_1, ...) if file already exists with that name.

    Args:
        fig (plt.fig): figure pointer
        ax (plt.ax): axis pointer
        name (str): desired name of file. No extension produced .png with name. Add .svg as ext. if needed
        dpi (int, optional): _description_. Defaults to 600.
    """
    # Set plot colours
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    ax.legend(
        fancybox=True,
        facecolor=(1.0, 1.0, 1.0, 1.0),
        labelcolor='black',
        loc='upper left',
        framealpha=0)
    ax.get_legend().remove()
    fig.tight_layout()

    # Loop to check for file - appends filename with _# if name already exists
    f_exist = True
    app_no = 0
    while f_exist is True:
        if os.path.exists(name + '.png') is False:
            ax.figure.savefig(name, dpi=dpi)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no), dpi=dpi)
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


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


def Gaussian_2D(coords, A, xo, yo, σ_x, σ_y, θ, offset):
    """2D plot on coord fields of Gaussian peak

    Args:
        coords (tuple of 2 arrays): coordinate grids, output of np.meshgrid
        A (float): peak value
        xo (float): x location of peak
        yo (float): y location of peak
        offset (background): constant z offset

    Returns:
        _type_: array of z values for Gaussian peak 
    """
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (offset + A * np.exp(- (a * ((x - xo) ** 2) +
                                2 * b * (x - xo) * (y - yo) +
                                c * ((y - yo) ** 2))))
    return g.ravel()


def Gaussian_1D(x, A, x_c, x_w, bkg=0, N=1):
    """ See wikipedia
    https://en.wikipedia.org/wiki/Normal_distribution

    Args:
        x (array): x values
        A (float): peak value
        x_c (float): mean
        x_w (float): standard deviation
        bkg (int, optional): y offset. Defaults to 0.
        N (int, optional): order. Defaults to 1.

    Returns:
        array: y values
    """
    # Note the optional input N, used for super Gaussians (default = 1)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * x_w ** 2))**N) + bkg
    return G


def cart2pol(x, y):
    """convert x y coords to rho phi coords

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
    """_summary_

    Args:
        rho (_type_): _description_
        phi (_type_): _description_

    Returns:
        x, y (tuple, (array, array)): x y coords
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def rbytheta(x, y, theta):
    """_summary_

    Args:
        x (array): x coords
        y (array): y coords
        theta (float): angular rotation (theta = 0 is positive y direction, positive = c.w.)

    Returns:
        (Rx, Ry) (tuple(array,array)): rotated x and y coords 
    """
    rho, phi = cart2pol(x, y)
    Rx, Ry = pol2cart(rho, phi + theta)
    return (Rx, Ry)


# %% stuff
# cavity parameters
L = 20e-6
R = 20e-6
d = 12e-6
wavelength_m = 780e-9

# initially work in mirror coords - (0,0) bottom centre of m0
# this is the tilt of mirror 1 w.r.t mirror 0
theta1 = 0.0

# coordinates of mirror 1 centre of RoC
r1_c = 0
z1_c = L-R

# coordinates of mirror 0 centre of RoC (x0_c must = 0)
r0_c = 0
z0_c = R

r1_c_act = L*np.tan(theta1) + r1_c

# simulation region and resolution
r_range = 0.8*R
z_range = 1.1*L
r_points = 1000
z_points = 2000

# height/depth of feature
h = R - np.sqrt(R**2 - (d**2/4))

# angular tilt associated with lateral offset
theta2 = np.arctan((r1_c-r0_c)/(z0_c-z1_c))

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

# Surface of mirror 0
# Full semi-circle rz
C0_z_semi = -np.sqrt(np.abs(R**2 - (rs-r0_c)**2)) + z0_c
# Flat surface at height h
S0 = h*np.ones_like(rs)
# x indices where semi-circle crosses surface
idx0 = np.argwhere(np.diff(np.sign(C0_z_semi - S0))).flatten()

# Cavity feature xy
C0_z = C0_z_semi[idx0[0]:idx0[-1]]
C0_r = rs[idx0[0]:idx0[-1]]

# Surface of mirror 1
C1_z_semi = np.sqrt(np.abs(R**2 - (rs-r1_c)**2)) + z1_c
# Flat surface at height L - h
S1 = (L - h)*np.ones_like(rs)
# x indices where semi-circle crosses surface
idx1 = np.argwhere(np.diff(np.sign(C1_z_semi - S1))).flatten()
# Convert cavity feature xy to r phi (shift to x=0 for rotation)
S1_r, S1_phi = cart2pol(rs[idx1[0]:idx1[-1]]-r1_c, C1_z_semi[idx1[0]:idx1[-1]])
# Rotate feature theta 1
S1_phi = S1_phi - theta1
# Reconvert cavity feature back to xy
C1_r, C1_z = pol2cart(S1_r, S1_phi)
# Shift back from x=0 after rotation
C1_r = C1_r + r1_c

# Generate some new regions based on the mirror feature locations
zs_0 = np.linspace(-h, 3*h, z_points)
zs_1 = np.linspace(np.min(C1_z) - h, h + np.max(C1_z), z_points)

# calc G00 mode - NOTE symmetric cavity assumed (RoC of faces are the same)
# Gaussian beam parameters (see wikipedia)
zR = np.sqrt((R/(L/2)-1))*(L/2)
n = np.floor(L/wavelength_m)
l = L/n
w0 = np.sqrt((zR * l) / (np.pi))
wm = w0 * np.sqrt(1+(L/(2*zR))**2)
kx = (zs-L/2)*np.tan(-theta2)+r1_c/2
print('beam waist = ', np.round(1e6*w0, 3), ' μm')
print('beam waist on mirror = ', np.round(1e6*wm, 3), ' μm')
# Coords normal to mirror 0
coords = np.meshgrid(zs, rs)
# Sub regions around each feature
coords0 = np.meshgrid(zs_0, rs)
coords1 = np.meshgrid(zs_1, rs)

# rotate each coordinate grid for irradiance calculations
Rcoords = rbytheta(coords[0], coords[1], theta2)
Rcoords0 = rbytheta(coords0[0], coords0[1], theta2)
Rcoords1 = rbytheta(coords1[0], coords1[1], theta2)

# calculate the irradiance in each rotated coord system
GI, GE = Gaussian_beam(Rcoords[0], Rcoords[1] - r1_c, (L/2), w0, l, 1)
GI_0, GE_0 = Gaussian_beam(Rcoords0[0], Rcoords0[1] - r1_c, (L/2), w0, l, 1)
GI_1, GE_1 = Gaussian_beam(Rcoords1[0], Rcoords1[1] - r1_c, (L/2), w0, l, 1)

RC0_z, RC0_r = rbytheta(C0_z, C0_r, theta2)
RC1_z, RC1_r = rbytheta(C1_z, C1_r, theta2)
GI_m0, _ = Gaussian_beam(RC0_z, RC0_r - r1_c, L/2, w0, l, 1)
GI_m1, _ = Gaussian_beam(RC1_z, RC1_r - r1_c, L/2, w0, l, 1)

popt0, cov0 = curve_fit(Gaussian_1D, C0_r, GI_m0,
                        p0=[np.max(GI_m0), r0_c, wm])

popt1, cov1 = curve_fit(Gaussian_1D, C1_r-r1_c_act, GI_m1,
                        p0=[np.max(GI_m0), -r1_c_act, wm])

dx0 = popt0[1]
dx1 = popt1[1]

I_0 = GI_0[:, np.argmin(np.abs(zs-0))]
I_L = GI_1[:, np.argmin(np.abs(zs-L))]
I_waist = GI[:, np.argmin(np.abs(zs-L/2))]

# PLOTS!
# this is to add new colormaps with some transparency to the list
ncolors = 256
GnBu_t0 = plt.get_cmap('GnBu')(range(ncolors))
GnBu_t0[:, -1] = np.linspace(0, 1, ncolors)
map_object = LinearSegmentedColormap.from_list(name='GnBu_t0', colors=GnBu_t0)

# full mode picture
ax1, fig1 = set_figure('mirror arrangement',
                       'r / μm',
                       'z / μm')

img = ax1.imshow(np.transpose(GI/np.max(GI)),
                 extent=extents(1e6*rs) + extents(1e6*zs),
                 #    norm=LogNorm(vmin=1e-6, vmax=1e6),
                 cmap=map_object,
                 origin='lower',
                 )
# img = ax1.imshow(np.transpose(G_alt),
#            extent=extents(1e6*xs)+extents(1e6*ys),
#         #    norm=LogNorm(vmin=1e-6, vmax=1e6),
#         cmap='magma',
#         origin='lower',
#            )
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig1.colorbar(img, cax=cax1)
cbar1.ax.get_yaxis().labelpad = 15
#
ax1.plot(1e6*kx, 1e6*zs, '-',
         lw=0.5,
         color='xkcd:black')
ax1.plot(0, 0,
         '.',
         color='xkcd:black')
ax1.plot(1e6*r1_c_act, 1e6*L,
         '.',
         color='xkcd:black')
ax1.plot(1e6*C0_r, 1e6*C0_z,
         '-',
         color='xkcd:red',
         label='m0')
ax1.plot(1e6*C1_r, 1e6*C1_z,
         '-',
         color='xkcd:blue',
         label='m1')
ax1.plot(1e6*r0_c, 1e6*z0_c,
         'x',
         color='xkcd:red',
         )
ax1.plot(1e6*r1_c, 1e6*(z1_c),
         'x',
         color='xkcd:blue',
         )

ax1.plot(1e6*rs[int(0.1*r_points):int(0.9*r_points)],
         1e6*C0_z_semi[int(0.1*r_points):int(0.9*r_points)],
         ':',
         color='xkcd:red',
         )
ax1.plot(1e6*rs,
         1e6*C1_z_semi,
         ':',
         color='xkcd:blue',
         )
ax1.set_aspect('equal')
ax1.legend()

# waist comparisons
ax2, fig2 = set_figure('profile on mirror', 'Δx / μm', 'I / a.u.')
ax2.plot(1e6*rs, I_waist,
         '-',
         alpha=0.1,
         color='xkcd:black',
         label='waist')
ax2.plot([0, 0], [0, np.max(I_waist)],
         ':',
         color='xkcd:black',
         alpha=0.1,
         )
ax2.plot(1e6*C0_r, GI_m0,
         '-',
         alpha=1,
         color='xkcd:red')
ax2.plot(1e6*rs, I_0,
         ':',
         alpha=1,
         color='xkcd:red')
ax2.plot(1e6*(C1_r-r1_c_act), GI_m1,
         '-',
         alpha=1,
         color='xkcd:blue')
ax2.plot(1e6*(rs-r1_c_act), I_L,
         ':',
         alpha=1,
         color='xkcd:blue')
ax2.text(2*dx0, 1.2*np.max(GI_m0),
         ('Δx$_0$ =' + str(np.round(1e6*dx0, 1))),
         color='xkcd:black')
ax2.text(2*dx1-L*np.tan(theta1), 1.05*np.max(GI_m1),
         ('Δx$_1$ =' + str(np.round(1e6*dx1, 1))),
         color='xkcd:black')
ax2.text(1e6*w0/2, 0.75*np.max(I_waist),
         ('confocal waist = ' + str(np.round(1e6*w0, 2)) + ' μm'),
         color='xkcd:black')
ax2.text(1e6*wm/2, 0.25*np.max(I_waist),
         ('waist on mirror = ' + str(np.round(1e6*wm, 2)) + ' μm'),
         color='xkcd:black')
ax2.plot([-1e6*d/2, -1e6*d/2], [0, 1], ':',
         color='xkcd:grey',
         alpha=0.5,
         )
ax2.plot([1e6*d/2, 1e6*d/2], [0, 1], ':',
         color='xkcd:grey',
         alpha=0.5,
         )
# ax2.set_yscale('log')

# Mirror 0 surface and mode
ax3, fig3 = set_figure('mirror 0',
                       'x / μm',
                       'y / μm')
img = ax3.imshow(np.transpose(GI_0),
                 extent=extents(1e6*rs)+extents(1e6*zs_0),
                 #    norm=LogNorm(vmin=1e-6, vmax=1e6),
                 cmap='viridis',
                 origin='lower',
                 )
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig3.colorbar(img, cax=cax3)
cbar3.ax.get_yaxis().labelpad = 15

ax3.plot(1e6*C0_r, 1e6*C0_z,
         '-',
         color='xkcd:red',
         label='m0')
# ax3.set_ylim([-h,3*h])
ax3.set_aspect('auto')

# Mirror 1 surface and mode
ax4, fig4 = set_figure('mirror 1',
                       'x / μm',
                       'y / μm')
img = ax4.imshow(np.transpose(GI_1),
                 extent=extents(1e6*rs)+extents(1e6*zs_1),
                 #    norm=LogNorm(vmin=1e-6, vmax=1e6),
                 cmap='viridis',
                 origin='lower',
                 )
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="5%", pad=0.05)
cbar4 = fig4.colorbar(img, cax=cax4)
cbar4.ax.get_yaxis().labelpad = 15
# ax4.plot(1e6*(xs),
#          1e6*C1,
#          ':',
#          color='xkcd:blue',
#          )
ax4.plot(1e6*C1_r, 1e6*C1_z,
         '-',
         color='xkcd:blue',
         )

# ax4.set_ylim([1e6*(np.min(C1_y) - h), 1e6*(h + np.max(C1_y))])
ax4.set_aspect('auto')

# % Plot 2d mode distribution
theta = np.linspace(0, 2 * np.pi, 150)
radius = d/2
a = radius * np.cos(theta)
b = radius * np.sin(theta)

M0 = Gaussian_2D(np.meshgrid(rs, rs), 1, dx0, 0, popt0[2], popt0[2], 0, 0)
M0 = np.reshape(M0, np.shape(np.meshgrid(rs, rs)[0]))
M1 = Gaussian_2D(np.meshgrid(rs, rs), 1, dx1, 0, popt1[2], popt1[2], 0, 0)
M1 = np.reshape(M1, np.shape(np.meshgrid(rs, rs)[0]))
ax5, fig5 = set_figure('Mirror 0 2d', 'x', 'y')
ax5.plot(a, b,
         color='xkcd:black')
ax5.set_xlim((-d, d))
ax5.set_ylim((-d, d))
ax6, fig6 = set_figure('Mirror 1 2d', 'x', 'y')
ax6.plot(a, b,
         color='xkcd:black')
ax5.plot(a, b,
         ':',
         color='xkcd:black')
ax6.set_xlim((-d, d))
ax6.set_ylim((-d, d))

A0 = np.sum(M0)
A1 = np.sum(M1)
Mask = np.ones_like(M0)
for i0, v0 in enumerate(rs):
    for i1, v1 in enumerate(rs):
        if v0**2 + v1**2 > radius ** 2:
            Mask[i0, i1] = 0

ax5.imshow(np.multiply(M0, Mask), extent=extents(rs)+extents(rs),
           cmap='Reds',
           alpha=1)

ax6.imshow(np.multiply(M1, Mask), extent=extents(rs)+extents(rs),
           cmap='Blues',
           alpha=1)
A0m = np.sum(np.multiply(M0, Mask))
A1m = np.sum(np.multiply(M1, Mask))
ax2.plot(1e6*rs, 0.5*np.multiply(M1, Mask)[500, :], color='xkcd:dark blue')
ax2.plot(1e6*rs, 0.5*np.multiply(M0, Mask)[500, :], color='xkcd:dark red')
# ax2.set_yscale('log')
ax2.set_ylim(1e-10, 1)
A_0_PPM_2d = 1e6*(1 - A0m/A0)
A_L_PPM_2d = 1e6*(1 - A1m/A1)
ax5.text(0, 0.7*d,
         (' loss = ' + str(np.round(A_0_PPM_2d, 1)) + ' ppm'))
ax6.text(0, 0.7*d,
         (' loss = ' + str(np.round(A_L_PPM_2d, 1)) + ' ppm'))
plt.show()

# %% save plots
os.chdir(r'G:\My Drive\Plots')
PPT_save_plot(fig1, ax1, 'mode.svg')
