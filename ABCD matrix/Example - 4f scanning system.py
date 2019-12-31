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
# w0 is the Gaussian waist of the starting beam
w0 = 5e-3
λ0 = 532e-9
# Optical path set-up [manual] ################################################
# lens details
f1 = 125e-3
f2 = 125e-3
f3 = 0.5e-3
fres = 100000

# Propagate the beam from FSM (z0) --> l1 (z1)
# Propagate the beam from l1 (zz) --> l2 (z2)
z0 = 0
z1 = f1
z2 = 2 * f1 + f2
z3 = 2 * (f1 + f2)
z4 = 2 * (f1 + f2) + 2 * f3

# Choose ref.indices
n0 = 1

# Calculated parameters
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
g0 = z0 + zR * 1j

x0 = 0
x1 = w0
x2 = -w0
θ0 = 0
θ1 = 0.1

qs0 = [[x0, θ0]]
qs1 = [[x1, θ0]]
qs2 = [[x2, θ0]]

ps0 = [[x0, θ1]]
ps1 = [[x1, θ1]]
ps2 = [[x2, θ1]]

Gs = [[g0, 1]]

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

# Propagation from z1 to z2 (low res towards focus)
zs2, qs0, ns2 = prd_tmat.ABCD_propagate(qs0, z2, zs1, ns1)
_, qs1, _ = prd_tmat.ABCD_propagate(qs1, z2, zs1, ns1)
_, qs2, _ = prd_tmat.ABCD_propagate(qs2, z2, zs1, ns1)

_, ps0, _ = prd_tmat.ABCD_propagate(ps0, z2, zs1, ns1)
_, ps1, _ = prd_tmat.ABCD_propagate(ps1, z2, zs1, ns1)
_, ps2, _ = prd_tmat.ABCD_propagate(ps2, z2, zs1, ns1)

_, Gs, _ = prd_tmat.ABCD_propagate(Gs, z2, zs1, ns1)

# Pass through 2nd thin lens
qs0 = prd_tmat.ABCD_tlens(qs0, f2)
qs1 = prd_tmat.ABCD_tlens(qs1, f2)
qs2 = prd_tmat.ABCD_tlens(qs2, f2)

ps0 = prd_tmat.ABCD_tlens(ps0, f2)
ps1 = prd_tmat.ABCD_tlens(ps1, f2)
ps2 = prd_tmat.ABCD_tlens(ps2, f2)

Gs = prd_tmat.ABCD_tlens(Gs, f2)

# Propagation from z2 to z3 (low res towards focus)
zs3, qs0, ns3 = prd_tmat.ABCD_propagate(qs0, z3, zs2, ns2)
_, qs1, _ = prd_tmat.ABCD_propagate(qs1, z3, zs2, ns2)
_, qs2, _ = prd_tmat.ABCD_propagate(qs2, z3, zs2, ns2)

_, ps0, _ = prd_tmat.ABCD_propagate(ps0, z3, zs2, ns2)
_, ps1, _ = prd_tmat.ABCD_propagate(ps1, z3, zs2, ns2)
_, ps2, _ = prd_tmat.ABCD_propagate(ps2, z3, zs2, ns2)

_, Gs, _ = prd_tmat.ABCD_propagate(Gs, z3, zs2, ns2)

# Pass through 3rd thin lens
qs0 = prd_tmat.ABCD_tlens(qs0, f3)
qs1 = prd_tmat.ABCD_tlens(qs1, f3)
qs2 = prd_tmat.ABCD_tlens(qs2, f3)

ps0 = prd_tmat.ABCD_tlens(ps0, f3)
ps1 = prd_tmat.ABCD_tlens(ps1, f3)
ps2 = prd_tmat.ABCD_tlens(ps2, f3)

Gs = prd_tmat.ABCD_tlens(Gs, f3)

# Propagation from z3 to z4 (low res towards focus)
zs4, qs0, ns4 = prd_tmat.ABCD_propagate(qs0, z4, zs3, ns3, fres)
_, qs1, _ = prd_tmat.ABCD_propagate(qs1, z4, zs3, ns3, fres)
_, qs2, _ = prd_tmat.ABCD_propagate(qs2, z4, zs3, ns3, fres)

_, ps0, _ = prd_tmat.ABCD_propagate(ps0, z4, zs3, ns3, fres)
_, ps1, _ = prd_tmat.ABCD_propagate(ps1, z4, zs3, ns3, fres)
_, ps2, _ = prd_tmat.ABCD_propagate(ps2, z4, zs3, ns3, fres)

_, Gs, _ = prd_tmat.ABCD_propagate(Gs, z4, zs3, ns3, fres)

##############################################################################
# Convert 1D arrays of qs, ps & ns into Rs, ws, xs and θs
##############################################################################
# Invert q parameter
gs_inv = 1 / np.array(Gs)[:, 0]
# Calculate Radii of curvature from inverted qs
Rs = 1 / np.real(gs_inv)
# Calculate waists from ns, inverted qs and λ0
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns4) * np.imag(gs_inv))))

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
plot_path = r"C:\local files\Python\Plots"
zs_plot = 1e0 * np.array(zs4)
qxs0 = 1e6 * qxs0
qxs1 = 1e6 * qxs1
qxs2 = 1e6 * qxs2
pxs0 = 1e6 * pxs0
pxs1 = 1e6 * pxs1
pxs2 = 1e6 * pxs2

ws = 1e6 * ws

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('optical axis (m)')
ax1.set_ylabel('y axis - beam waist (μm)')

plt.plot(zs_plot, qxs0, '-', c=cs['ggdred'],
         alpha=0.5, label='on axis')
plt.plot(zs_plot, pxs0, '-', c=cs['ggdblue'],
         alpha=0.5, label='Δλ = 3 nm deflected')

plt.plot(zs_plot, qxs1, '-', c=cs['ggred'], label='')
plt.plot(zs_plot, qxs2, '-', c=cs['ggred'], label='')
plt.plot(zs_plot, pxs1, '-', c=cs['ggblue'], label='')
plt.plot(zs_plot, pxs2, '-', c=cs['ggblue'], label='')

plt.plot(zs_plot, ws, '--', c=cs['gglred'], label='Gaussian Beam')
plt.plot(zs_plot, -ws, '--', c=cs['gglred'])

ax1.legend(loc='upper right', fancybox=True, framealpha=1)
ax1.set_xlim(2.0 * (f1 + f1) + 0.99 * f3, 2.0 * (f1 + f1) + 2 * f3 - 0.99 * f3)
ax1.set_ylim(-250, 250)
plt.tight_layout()
plt.show()

# Saving plots
plot_file_name = plot_path + r'\Gauss and Ray tracing off axis.png'
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
