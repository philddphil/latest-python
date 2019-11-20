##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
p_lib = (r"C:\local files\Python\Local Repo\library")
p_lib = (r"C:\GitHub\latest-python\library")
sys.path.insert(0, p_lib)

import prd_plots
import prd_tmat
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################

# Follows the wikipedia ray matrix formalism
# (scroll down for the Gaussian beam bit)

# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# p is  a normal ray trace

# User defined parameters
w0 = 2e-4
λ0 = 633e-9

# Some handy refractive indices
n0 = 1
n1 = 1.44

# Optical path set-up [manual] ################################################
# Beam goes from z0 to lens at z1
# Then beam is focussed to a waist, and model terminates at z2

# First thin lens details location and focal length
z0 = 0
z1 = 45e-3
f1 = 45e-3

# Finishing point of model
z2 = 100e-3

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
x0 = w0
θ0 = 0
q0 = z0 + zR * 1j
qs = [[q0, 1]]
ps = [[x0, θ0]]
zs0 = [0]
ns0 = [1]

# Propagation from z0 to z1
zs1, qs, ns1 = prd_tmat.ABCD_propagate(qs, z1)
_, ps, _ = prd_tmat.ABCD_propagate(ps, z1)

# Pass through 1st thin lens
qs = prd_tmat.ABCD_tlens(qs, f1)
ps = prd_tmat.ABCD_tlens(ps, f1)

# Propagation from thin lens ==> end of region
zs2, qs, ns2 = prd_tmat.ABCD_propagate(qs, z2, zs1, ns1)
_, ps, _ = prd_tmat.ABCD_propagate(ps, z2, zs1, ns1)


##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
qs_inv = 1 / np.array(qs)[:, 0]
# Calculate Radii of curvature from inverted qs
Rs = 1 / np.real(qs_inv)
# Calculate waists from ns, inverted qs and λ0
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns2) * np.imag(qs_inv))))
# xs are the ray tracing equivalent to the beam waists
xs = np.array(ps)[:, 0]
# θs are the divergence angles of the beam at each point
θs = np.array(ps)[:, 1]

##############################################################################
# Plot the outputted waists
##############################################################################
# Scale values for appropriate plotting
prd_plots.ggplot()
plot_path = r"C:\local files\Python\Plots"
zs = 1e0 * np.array(zs2)
ws = 1e3 * ws
xs = 1e3 * xs

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (m)')
ax1.set_ylabel('y axis - beam waist (mm)')
plt.plot(zs, ws, '.-', c=cs['ggred'], label='Gaussian Beam')
plt.plot(zs, -ws, '.-', c=cs['ggred'])
plt.plot(zs, xs, '-', c=cs['ggblue'], label='Raytrace')
plt.plot(zs, -xs, '-', c=cs['ggblue'])
# plt.plot([1e6 * z2, 1e6 * z2],  [220, -
#                                  220], '-', c=cs['mdk_orange'])
plt.plot([z1, z1], [np.max(ws), - np.max(ws)],
         '-', c=cs['mnk_yellow'], alpha=0.5)
# plt.plot(exp_x, 2*exp_y, '.:', c=cs['mdk_orange'])
# plt.plot(exp_x, -2*exp_y, '.:', c=cs['mdk_orange'])
# plt.plot(1e6 * zs, c=cs['mdk_pink'])
# fig2 = plt.figure('fig2')
# ax2 = fig2.add_subplot(1, 1, 1)
# fig2.patch.set_facecolor(cs['mdk_dgrey'])
# ax2.set_xlabel('optical axis (μm)')
# ax2.set_ylabel('ray angle Θ (degrees)')
# plt.plot(1e6 * zs, 180 * θs / π, c=cs['ggred'])
plt.tight_layout()
plt.show()
plot_file_name = plot_path + r'\Gauss and Ray tracing foucssing collimated.png'
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
