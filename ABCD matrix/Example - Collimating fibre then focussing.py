# %% Import some libraries
import os
import sys
import codecs
import copy
import numpy as np
import matplotlib.pyplot as plt

# %% defs


def ABCD_MM(q_in, d, n=1):
    """ Matrix multiplication step in propagating 'd' along optical axis

    Args:
        q_in (list): list of 2 floats parameterising ray/Gbeam
        d (float): float, distance to propagate
        n (int, optional): Refractive index of medium. Defaults to 1.
    """
    M = np.array([[1, d * n], [0, 1]])
    q_out = np.matmul(M, q_in)
    return (q_out)


def ABCD_propagate(qs, z_end, zs_in=None, ns_in=None, res=1000):
    """_summary_

    Args:
        qs (list): 2 element list of floats parameterising ray/Gbeam
        z_end (float): end of region to propagate over
        zs_in (list, optional): list of locations (floats) ray/Gbeam evaluated at. Defaults to None.
        ns_in (list, optional): list ref inds at locations (floats) ray/beam evaluated at. Defaults to None.
        res (int, optional): int number of points to evaluate ray/Gbeam at. Defaults to 1000.
    """
    if zs_in is None:
        zs_in = [0]
    if ns_in is None:
        ns_in = [1]
    zs_out = copy.copy(zs_in)
    qz = qs
    q0 = qs[-1]
    z_start = zs_in[-1]
    zs_i = np.linspace(z_start, z_end, res)
    ns = ns_in[-1] * np.ones(len(zs_i))
    ns_out = copy.copy(ns_in)
    if q0[1] == 1:
        z_start = np.real(q0[0])

    dz = zs_i[1] - zs_i[0]

    for i1, val1 in enumerate(zs_i[0:]):
        q1 = ABCD_MM(q0, dz, ns[i1])
        qz.append(q1)
        q0 = q1
        zs_out.append(zs_i[i1])
        ns_out.append(ns[i1])

    return (zs_out, qz, ns_out)


def ABCD_tlens(qs, f):
    """ Passes ray/Gbeam through thin lens, focal length l 

    Args:
        qs (list): 2 element list of floats parameterising ray/Gbeam
        f (float): focal length of lens

    Returns:
        _type_: _description_
    """
    M = np.array([[1, 0], [-1 / f, 1]])
    q_out = np.matmul(M, qs[-1])
    if qs[-1][1] == 1:
        q_out = q_out / q_out[1]
    qs[-1] = q_out
    return qs


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


# %% Do some stuff
# Follows the wikipedia ray matrix formalism
# (scroll down for the Gaussian beam bit)

# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# p is  a normal ray trace

# units are m

# User defined parameters
w0 = 4.4e-6 / 2
λ0 = 780e-9

# Some handy refractive indices
n0 = 1
n1 = 1.44

# %% Optical path set-up [manual]
# # Beam goes from z0 to lens at z1
# Then beam is focussed to a waist, and model terminates at z2

# Location of w0
z0 = 0

# First thin lens details location and focal length
z1 = 18.4e-3
f1 = 18.4e-3

# Location of second thin lens and focal length
z2 = 2 * z1
f2 = 50e-3

# Propagate to focal region
z3 = z2 + 49e-3

# End of propagation
z4 = z2 + 51e-3

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
x0 = w0
θ0 = λ0 / (np.pi * n0 * w0)
q0 = z0 + zR * 1j
qs = [[q0, 1]]
ps = [[x0, θ0]]
zs0 = [0]
ns0 = [1]

# Propagation from z0 to z1
zs1, qs, ns1 = ABCD_propagate(qs, z1)
_, ps, _ = ABCD_propagate(ps, z1)

# Pass through 1st thin lens
qs = ABCD_tlens(qs, f1)
ps = ABCD_tlens(ps, f1)

# Propagation from 1st thin lens ==> 2nd thin lens
zs2, qs, ns2 = ABCD_propagate(qs, z2, zs1, ns1)
_, ps, _ = ABCD_propagate(ps, z2, zs1, ns1)

# Pass through 2nd thin lens
qs = ABCD_tlens(qs, f2)
ps = ABCD_tlens(ps, f2)

# Propagation from 2nd thin lens to focal region
zs3, qs, ns3 = ABCD_propagate(qs, z3, zs2, ns2)
_, ps, _ = ABCD_propagate(ps, z3, zs2, ns2)

# Propagation through focal region to end of simulation
zs4, qs, ns4 = ABCD_propagate(qs, z4, zs3, ns3)
_, ps, _ = ABCD_propagate(ps, z4, zs3, ns3)

##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
qs_inv = 1 / np.array(qs)[:, 0]
# Calculate Radii of curvature from inverted qs
Rs = 1 / np.real(qs_inv)
# Calculate waists from ns, inverted qs and λ0
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns4) * np.imag(qs_inv))))
# xs are the ray tracing equivalent to the beam waists
xs = np.array(ps)[:, 0]
# θs are the divergence angles of the beam at each point
θs = np.array(ps)[:, 1]

##############################################################################
# Calculate FOM
##############################################################################

zs = 1e0 * np.array(zs4)
ws = 1e3 * ws
xs = 1e3 * xs

# R1 = [0:999]
# R2 = [1000:1999]
# R3 = [2000:2999]
# R4 = [3000:3999]

w_min = 2 * np.round(1e3 * np.min(ws[3000:3999]), 3)
w_max = 2 * np.round(1e3 * np.max(ws[3000:3999]), 3)
θ_f = 1e6 * λ0 / (π * w_min)
print('Min diameter at focus (FWHM) = ', w_min, 'μm')

print('N.A. of Gaussian from min w = ', θ_f)

##############################################################################
# %% Plot the outputted waists
# Scale values for appropriate plotting
ax1, fig1 = set_figure('plot','optical axis / m', 'y axis', 3)

plt.plot(zs, ws, 
         '.-', 
         c='xkcd:red', 
         label='Gaussian Beam')
plt.plot(zs, -ws, 
         '.-', 
         c='xkcd:red')

plt.plot(zs, xs, '-', 
         c='xkcd:blue', 
         label='Raytrace')
plt.plot(zs, -xs, 
         '-', 
         c='xkcd:blue')

plt.plot([z1, z1], [np.max(ws), - np.max(ws)],
         '-', 
         c='xkcd:yellow', 
         alpha=0.5)

plt.plot([z2, z2], [np.max(ws), - np.max(ws)],
         '-', 
         c='xkcd:yellow', 
         alpha=0.5)

# ax1.set_xlim(z3, z4)
# ax1.set_ylim(-0.05, 0.05)

plt.tight_layout()
plt.show()

# Saving plots

ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
# prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
