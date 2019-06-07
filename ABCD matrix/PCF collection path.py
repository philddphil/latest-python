##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
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
w0 = 2e-3
λ0 = 633e-9

# Some handy refractive indices
n0 = 1
n1 = 1.44

# Optical path set-up [manual] ################################################

# First thin lens details:
z1 = 45e-3
f1 = 45e-3

# Photonic crystal fibre MFD
PCF = 7e-6
# Start of pin hole region:
z2 = z1 + f1 - 2e-3

# End of pin hole region:
z3 = z1 + f1 + 2e-3

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
z0 = 0
x0 = w0
θ0 = λ0 / (π * w0)
p0 = np.array([x0, θ0])
q0 = np.array([z0 + zR * 1j, 1])
ps = np.empty([0, 2])
qs = np.empty([0, 2])
zs = np.empty([0])
ns = np.empty([0])

# N.A of fibre
print('N.A = ', np.round(np.sin(θ0), 3))
print('zR = ', np.round(1e6 * zR, 3))
print('q0 = ', np.round(q0, 3))

# Propagation: fibre ==> 1st thin lens
zs0, qs0, ns0 = prd_tmat.ABCD_propagate(q0, z1)
_, ps0, _ = prd_tmat.ABCD_propagate(p0, z1)
qs = np.append(qs, qs0, axis=0)
ps = np.append(ps, ps0, axis=0)
zs = np.append(zs, zs0, axis=0)
ns = np.append(ns, ns0, axis=0)

# Pass through 1st thin lens
q1 = prd_tmat.ABCD_tlens(qs[-1], f1)
p1 = prd_tmat.ABCD_tlens(ps[-1], f1)

# Propagation: thin lens ==>  start of pin hole region
zs1, qs1, ns1 = prd_tmat.ABCD_propagate(q1, z2, z_start=z1)
_, ps1, _ = prd_tmat.ABCD_propagate(p1, z2, z_start=z1)
qs = np.append(qs, qs1, axis=0)
ps = np.append(ps, ps1, axis=0)
zs = np.append(zs, zs1, axis=0)
ns = np.append(ns, ns1, axis=0)

# Propagation: start ==> end of pin hole region
zs2, qs2, ns2 = prd_tmat.ABCD_propagate(qs[-1], z3, z_start=z2, res=10000)
_, ps2, _ = prd_tmat.ABCD_propagate(ps[-1], z3, z_start=z2, res=10000)
qs = np.append(qs, qs2, axis=0)
ps = np.append(ps, ps2, axis=0)
zs = np.append(zs, zs2, axis=0)
ns = np.append(ns, ns2, axis=0)

##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
qs_inv = 1 / np.array(qs)[:, 0]
# Calculate Radii of curvature from inverted qs
Rs = 1 / np.real(qs_inv)
# Calculate waists from ns, inverted qs and λ0
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
zs = 1e0 * zs
ws = 1e3 * ws
xs = 1e3 * xs
PCF = 1e3 * PCF

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
plt.plot([z1 + f1, z1 + f1], [np.max(ws), (PCF / 2)],
         'o-', c=cs['mnk_pink'], alpha=1)
plt.plot([z1 + f1, z1 + f1], [(PCF / 2) * -1, -np.max(ws)],
         'o-', c=cs['mnk_pink'], alpha=1)
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
prd_plots.PPT_save_2d(fig1, ax1, 'SMF output - Raytrace, G. Beam.png')
