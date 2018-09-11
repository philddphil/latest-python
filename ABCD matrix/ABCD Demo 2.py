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
# files1 = glob.glob('*.csv')
# exp_x = np.genfromtxt(files1[0])
# exp_y = np.genfromtxt(files1[1])

# Follows the wikipedia ray matrix formalism
# (scroll down for the Gaussian beam bit)

# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# definte initial q

w0 = 10.0e-6
λ0 = 633e-9
n0 = 1
π = np.pi
R0 = np.inf
zR = (π * w0**2) / λ0
z0 = 0
x0 = 0
θ0 = λ0 / (π * w0)
p0 = np.array([x0, θ0])
q0 = np.array([z0 + zR * 1j, 1])
ps = np.empty([0, 2])
qs = np.empty([0, 2])
zs = np.empty([0])
ns = np.empty([0])

# print('zR', np.round(1e6 * zR, 2))
print('N.A = ', np.round(np.sin(θ0),3))

n1 = 1.44
R1 = -321e-6
R2 = -4000e-6

z1 = 100e-6
z2 = z1 + 950e-6
z3 = z2 + 7080e-6
z4 = z3 + 1010e-6
z5 = 0.15

# Propagation fibre ==> Ls (planar surface)
zs0, qs0, ns0 = prd.ABCD_propagate(q0, z1, res=20, n=n0)
_, ps0, _ = prd.ABCD_propagate(p0, z1, res=20)
qs = np.append(qs, qs0, axis=0)
ps = np.append(ps, ps0, axis=0)
zs = np.append(zs, zs0, axis=0)
ns = np.append(ns, ns0, axis=0)

# Planar Ls surface
q1 = prd.ABCD_plan(qs[-1], n0, n1)
p1 = prd.ABCD_plan(ps[-1], n0, n1)

# Propagation Ls (planar) ==> Ls (curved surface)
zs1, qs1, ns1 = prd.ABCD_propagate(q1, z2, z_start=z1, res=20, n=n1)
_, ps1, _ = prd.ABCD_propagate(p1, z2, z_start=z1, res=20, n=n1)
qs = np.append(qs, qs1, axis=0)
ps = np.append(ps, ps1, axis=0)
zs = np.append(zs, zs1, axis=0)
ns = np.append(ns, ns1, axis=0)

# Curved Ls surface
q2 = prd.ABCD_curv(qs[-1], n1, n0, R1)
p2 = prd.ABCD_curv(ps[-1], n1, n0, R1)

# Propagation Ls (curved) ==> La (planar surface)
zs2, qs2, ns2 = prd.ABCD_propagate(q2, z3, z_start=z2, res=20)
_, ps2, _ = prd.ABCD_propagate(p2, z3, z_start=z2, res=20)
qs = np.append(qs, qs2, axis=0)
ps = np.append(ps, ps2, axis=0)
zs = np.append(zs, zs2, axis=0)
ns = np.append(ns, ns2, axis=0)

# Planar La planar surface
q3 = prd.ABCD_plan(qs[-1], n0, n1)
p3 = prd.ABCD_plan(ps[-1], n0, n1)

# Propagation La (planar) ==> La (curved surface)
zs3, qs3, ns3 = prd.ABCD_propagate(q3, z4, z_start=z3, res=20, n=n1)
_, ps3, _ = prd.ABCD_propagate(p3, z4, z_start=z3, res=20, n=n1)
qs = np.append(qs, qs3, axis=0)
ps = np.append(ps, ps3, axis=0)
zs = np.append(zs, zs3, axis=0)
ns = np.append(ns, ns3, axis=0)

# Curved La surface
q4 = prd.ABCD_curv(qs[-1], n1, n0, R2)
p4 = prd.ABCD_curv(ps[-1], n1, n0, R2)

# Propagation La (curved) ==> P0
zs4, qs4, ns4 = prd.ABCD_propagate(q4, z5, z_start=z4, res=200)
_, ps4, _ = prd.ABCD_propagate(p4, z5, z_start=z4, res=200)
qs = np.append(qs, qs4, axis=0)
ps = np.append(ps, ps4, axis=0)
zs = np.append(zs, zs4, axis=0)
ns = np.append(ns, ns4, axis=0)

qs_inv = 1 / np.array(qs)[:, 0]
Rs = 1 / np.real(qs_inv)
ws = np.sqrt(np.abs(λ0 / (π * np.array(ns) * np.imag(qs_inv))))
xs = np.array(ps)[:, 0]
θs = np.array(ps)[:, 1]

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('optical axis (mm)')
ax1.set_ylabel('y axis - beam waist (μm)')
# ax1.set_aspect(0.02)
# plt.plot(z0, w0, 'x', c=cs['ggred'])
# plt.plot(z1, w1_a, 'x', c=cs['ggred'])
# plt.plot(z3, w3_a, 'x', c=cs['ggred'])
# plt.plot(all_zs, all_ws, '.-', c=cs['ggred'])
# plt.plot(all_zs, -1 * all_ws, '.-', c=cs['ggred'])

plt.plot(1e3 * zs, 1e6 * ws, '-', c=cs['ggred'], label='Gaussian Beam')
plt.plot(1e3 * zs, -1e6 * ws, '-', c=cs['ggred'])
plt.plot(1e3 * zs, 1e6 * xs, '-', c=cs['ggblue'], label='Raytrace')
plt.plot(1e3 * zs, -1e6 * xs, '-', c=cs['ggblue'])
plt.plot(1e3 * zs, 4e1 * (ns-1), '-', c=cs['mdk_purple'])

# plt.plot([1e6 * z2, 1e6 * z2],  [220, -
#                                  220], '-', c=cs['mdk_orange'])
plt.plot([18, 18],  [1e6 * np.max(ws) / 2, -
                     1e6 * np.max(ws) / 2], '-', c=cs['mdk_pink'])
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

prd.PPT_save_2d(fig1, ax1, 'SMF output - Raytrace, G. Beam.png')
