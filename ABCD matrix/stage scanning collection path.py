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
np.set_printoptions(suppress=True, precision=3)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

# Follows the wikipedia ray matrix formalism
# (scroll down for the Gaussian beam bit)
# units are m
# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# p is  a normal ray trace

# Some handy refractive indices
n0 = 1
n1 = 1.44

# Optical path set-up [manual] ################################################
# The beam waist (at z = 0) and the wavelength
w0 = 2e-3
λ0 = 633e-9

# First thin lens details (z = location, f = focal length)
z1 = 2
f1 = 20e-3

# MM core diameter
MM = 50e-6

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

# Propagation: lens ==> 1st thin lens
zs0, qs0, ns0 = prd.ABCD_propagate(q0, z1)
_, ps0, _ = prd.ABCD_propagate(p0, z1)
qs = np.append(qs, qs0, axis=0)
ps = np.append(ps, ps0, axis=0)
zs = np.append(zs, zs0, axis=0)
ns = np.append(ns, ns0, axis=0)

# Pass through 1st thin lens
q1 = prd.ABCD_tlens(qs[-1], f1)
p1 = prd.ABCD_tlens(ps[-1], f1)

# Propagation: thin lens ==>  start of pin hole region
zs1, qs1, ns1 = prd.ABCD_propagate(q1, z2, z_start=z1)
_, ps1, _ = prd.ABCD_propagate(p1, z2, z_start=z1)
qs = np.append(qs, qs1, axis=0)
ps = np.append(ps, ps1, axis=0)
zs = np.append(zs, zs1, axis=0)
ns = np.append(ns, ns1, axis=0)

# Propagation: start ==> end of pin hole region
zs2, qs2, ns2 = prd.ABCD_propagate(qs[-1], z3, z_start=z2, res=10000)
_, ps2, _ = prd.ABCD_propagate(ps[-1], z3, z_start=z2, res=10000)
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
zs = 1e0 * zs
ws = 1e3 * ws
xs = 1e3 * xs
MM = 1e3 * MM

fig1 = plt.figure('fig1', figsize=(2, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (m)')
ax1.set_ylabel('y axis - beam waist (mm)')
plt.plot(zs, ws, '-', c=cs['ggred'], label='Gaussian Beam')
plt.plot(zs, -ws, '-', c=cs['ggred'])
plt.plot(zs, xs, '-', c=cs['ggblue'], lw=0.5, label='Raytrace')
plt.plot(zs, -xs, '-', c=cs['ggblue'], lw=0.5)
# plt.plot([1e6 * z2, 1e6 * z2],  [220, -
#                                  220], '-', c=cs['mdk_orange'])
plt.plot([z1, z1], [2 * np.max(ws), - 2 * np.max(ws)],
         '-', c=cs['mnk_yellow'], alpha=0.5, label='lens')
plt.plot([z1 + f1, z1 + f1], [2 * np.max(ws), (MM / 2)],
         '.-', c=cs['mnk_pink'], alpha=0.5, label='fibre')
plt.plot([z1 + f1, z1 + f1], [(MM / 2) * -1, -2 * np.max(ws)],
         '.-', c=cs['mnk_pink'], alpha=0.5)
# ax1.legend(loc='center left', fancybox=True, framealpha=1)
plt.xlim(2.0199, 2.0201)
plt.ylim(-1 * MM, MM)
plt.tight_layout()
plt.show()
# ax1.legend(loc='center left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig1, ax1, 'SMF output - Raytrace, G. Beam.png')
