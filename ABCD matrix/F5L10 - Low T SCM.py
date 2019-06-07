##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

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

# User defined parameters
# w0 is the Gaussian waist of the starting beam
w0 = 2e-6
λ0 = 935e-9
# Choose ref.indices
n0 = 1
##############################################################################
# Optical path set-up [manual] ###############################################

# Lens details ###############################################################
# Objective lens
f1 = 20e-3
# PCF coupling in lens
f2 = 50e-3
# PCF coupling out lens
f3 = 50e-3

# Propagation ################################################################ 
# Sample, z0 
z0 = 0
# Objective lens, z1
z1 = f1
# Collimated path on scanning stage
z2 = z1 + 50e-3
# Focussing into PCF
z3 = z2 + f2
# Collimating out of PCF
z4 = z3 + f3
# 1st mirror towards HBTs
z5 = z4 + 1
# 2nd mirror towards HBTs
z6 = z5 + 0.5
# Towards Lens to SPAD SMP
z7 = z6 + 1

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
g0 = z0 + zR * 1j
# Gaussian propagator
Gs = [[g0, 1]]

x0 = 0
x1 = 2e-3
x2 = -2e-3
θ0 = 0
θ1 = 3 * 4.462e-5
qs0 = [[x0, θ0]]
qs1 = [[x1, θ0]]
qs2 = [[x2, θ0]]
ps0 = [[x0, θ1]]
ps1 = [[x1, θ1]]
ps2 = [[x2, θ1]]
zs = [z0]
ns = [n0]

# Propagation from z0 to z1
zs1, qs0, ns1 = prd_tmat.ABCD_propagate(qs0, z1)
_, qs1, _ = prd_tmat.ABCD_propagate(qs1, z1)
_, qs2, _ = prd_tmat.ABCD_propagate(qs2, z1)

_, ps0, _ = prd_tmat.ABCD_propagate(ps0, z1)
_, ps1, _ = prd_tmat.ABCD_propagate(ps1, z1)
_, ps2, _ = prd_tmat.ABCD_propagate(ps2, z1)

_, Gs, _ = prd_tmat.ABCD_propagate(Gs, z1)

# Pass through 1st thin lens
qs0 = prd_tmat.ABCD_tlens(qs0, f1)
qs1 = prd_tmat.ABCD_tlens(qs1, f1)
qs2 = prd_tmat.ABCD_tlens(qs2, f1)

ps0 = prd_tmat.ABCD_tlens(ps0, f1)
ps1 = prd_tmat.ABCD_tlens(ps1, f1)
ps2 = prd_tmat.ABCD_tlens(ps2, f1)

Gs = prd_tmat.ABCD_tlens(Gs, f1)

# Propagation from z1 to z2
zs2, qs0, ns2 = prd_tmat.ABCD_propagate(qs0, z2, zs1, ns1)
_, qs1, _ = prd_tmat.ABCD_propagate(qs1, z2, zs1, ns1)
_, qs2, _ = prd_tmat.ABCD_propagate(qs2, z2, zs1, ns1)

_, ps0, _ = prd_tmat.ABCD_propagate(ps0, z2, zs1, ns1)
_, ps1, _ = prd_tmat.ABCD_propagate(ps1, z2, zs1, ns1)
_, ps2, _ = prd_tmat.ABCD_propagate(ps2, z2, zs1, ns1)

_, Gs, _ = prd_tmat.ABCD_propagate(Gs, z2, zs1, ns1)

##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
gs_inv = 1 / np.array(Gs)[:, 0]
# Calculate Radii of curvature from inverted qs
Rs = 1 / np.real(gs_inv)
# Calculate waists from ns, inverted qs and λ0
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns2) * np.imag(gs_inv))))

# xs are the ray tracing equivalent to the beam waists
qxs0 = np.array(qs0)[:, 0]
qxs1 = np.array(qs1)[:, 0]
qxs2 = np.array(qs2)[:, 0]

pxs0 = np.array(ps0)[:, 0]
pxs1 = np.array(ps1)[:, 0]
pxs2 = np.array(ps2)[:, 0]
# θs are the divergence angles of the beam at each point
qθs0 = np.array(qs0)[:, 1]
qθs1 = np.array(qs1)[:, 1]
qθs2 = np.array(qs2)[:, 1]

pθs0 = np.array(ps0)[:, 1]
pθs1 = np.array(ps1)[:, 1]
pθs2 = np.array(ps2)[:, 1]

##############################################################################
# Plot the outputted waists
##############################################################################
# Scale values for appropriate plotting
prd_plots.ggplot()
plot_path = r"D:\Python\Plots"
zs2 = 1e0 * np.array(zs2)
qxs0 = 1e3 * qxs0
qxs1 = 1e3 * qxs1
qxs2 = 1e3 * qxs2
pxs0 = 1e3 * pxs0
pxs1 = 1e3 * pxs1
pxs2 = 1e3 * pxs2

ws = 1e3 * ws

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (m)')
ax1.set_ylabel('y axis - beam waist (mm)')
plt.plot(zs2, qxs0, '-', c=cs['ggred'], label='on axis')
plt.plot(zs2, pxs0, '-', c=cs['ggblue'], label='Δλ = 3 nm deflected')

plt.plot(zs2, qxs1, '-', c=cs['ggdred'], label='')
plt.plot(zs2, qxs2, '-', c=cs['ggdred'], label='')
plt.plot(zs2, pxs1, '-', c=cs['ggdblue'], label='')
plt.plot(zs2, pxs2, '-', c=cs['ggdblue'], label='')

plt.plot(zs2, ws, '.-', c=cs['gglred'], label='Gaussian Beam')
plt.plot(zs2, -ws, '.-', c=cs['gglred'])

ax1.legend(loc='upper right', fancybox=True, framealpha=1)
ax1.set_xlim(2.21994, 2.22)
ax1.set_ylim(-0.01, 0.01)
plt.tight_layout()
plt.show()

# Saving plots
plot_file_name = plot_path + r'\Gauss and Ray tracing.png'
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
