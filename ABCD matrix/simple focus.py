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
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

# Follows the wikipedia ray matrix formalism
# (scroll down for the Gaussian beam bit)

# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# p is  a normal ray trace

# User defined parameters
fibre_MFD = 1e-6
w0 = fibre_MFD / 2
λ0 = 633e-9

# Some handy refractive indices
n0 = 1
n1 = 1.44

# Optical path set-up [manual] ################################################

# First thin lens details:
z1 = 40e-6
f1 = 20e-6

# Second thin lens details:
z2 = z1 + 5
f2 = 50e-3

# Start of pin hole region:
z3 = z2 + f2 - 2e-3

# End of pin hole region:
z4 = z2 + f2 + 2e-3

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
z0 = -20e-6
x0 = 0
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
zs0, qs0, ns0 = prd.ABCD_propagate(q0, z1)
_, ps0, _ = prd.ABCD_propagate(p0, z1)
qs = np.append(qs, qs0, axis=0)
ps = np.append(ps, ps0, axis=0)
zs = np.append(zs, zs0, axis=0)
ns = np.append(ns, ns0, axis=0)


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
prd.ggplot()
fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (m)')
ax1.set_ylabel('y axis - beam waist (mm)')
plt.plot(zs, ws, '-', c=cs['ggred'], label='Gaussian Beam')
plt.plot(zs, -ws, '-', c=cs['ggred'])

plt.tight_layout()
plt.axis('off')
plt.savefig('test.svg')
plt.show()
prd.PPT_save_2d(fig1, ax1, 'SMF output - Raytrace, G. Beam.png')
