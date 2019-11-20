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

# User defined parameters
# w0 is the Gaussian waist of the starting beam
w0 = 1e-2
λ0 = 1550e-9

# Optical path set-up [manual] ################################################
# Propagate the beam from z0 --> z1
z0 = 0
z1 = 700e3

# Choose ref.indices
n0 = 1

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
x0 = 0
θ0 = λ0 / (π * w0)
q0 = z0 + zR * 1j
qs = [[q0, 1]]
ps = [[x0, θ0]]
zs = [z0]
ns = [n0]

# Propagation from z0 to z1
zs, qs, ns = prd_tmat.ABCD_propagate(qs, z1)
_, ps, _ = prd_tmat.ABCD_propagate(ps, z1)

##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
qs_inv = 1 / np.array(qs)[:, 0]
# Calculate Radii (Rs) of curvature from inverted qs
Rs = 1 / np.real(qs_inv)
# Calculate waists (ws) from ns, inverted qs and λ0
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns) * np.imag(qs_inv))))
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
zs = 1e-3 * np.array(zs)
ws = 1e0 * ws
xs = 1e0 * xs

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (km)')
ax1.set_ylabel('y axis - beam waist (m)')
plt.plot(zs, ws, '-', c=cs['ggred'], label='Gaussian Beam')
plt.plot(zs, -ws, '-', c=cs['ggred'])
plt.plot(zs, xs, '-', c=cs['ggblue'], label='Raytrace')
plt.plot(zs, -xs, '-', c=cs['ggblue'])
ax1.legend(loc='upper right', fancybox=True, framealpha=1)
plt.tight_layout()
plt.show()

# Saving plots
plot_file_name = plot_path + r'\Gauss and Ray tracing Propagation.png'
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
