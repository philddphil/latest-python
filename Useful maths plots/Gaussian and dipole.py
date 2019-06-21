##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_maths

cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = r"path line 1"\
    r"path line 2"
NA0 = 0.4
NA1 = 0.7
λ0 = 0.532
λ_fl = 0.7
L = 0.2
r = np.linspace(-15, 15, 1000)
z = np.linspace(-20, 20, 1000)

w00 = λ0 / (np.pi * NA0)
w01 = λ0 / (np.pi * NA1)
coords = np.meshgrid(z, r)

I_rz, w_z0 = prd_maths.Gaussian_beam(*coords, w00, λ0)
I_rz1, w_z1 = prd_maths.Gaussian_beam(*coords, w01, λ0)

z_lim = 1
for i0, j0 in enumerate(I_rz):
    for i1, j1 in enumerate(j0):
        if j1 >= z_lim:
            I_rz[i0, i1] = z_lim

S_rz = prd_maths.Dipole_2D(*coords, L, λ_fl)
z_lim = 0.005
for i0, j0 in enumerate(S_rz):
    for i1, j1 in enumerate(j0):
        if j1 >= z_lim:
            S_rz[i0, i1] = z_lim

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"

###### image plot ############################################################
fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis (μm)')
ax1.set_ylabel('y axis (μm)')
ax1.set_ylim(-15, 15)
ax1.set_xlim(-20, 0)
# plt.imshow(z_lim - I_rz, extent=prd_plots.extents(z) + prd_plots.extents(r),
#            cmap='Greens')

# plt.imshow(0.1 * S_rz, extent=prd_plots.extents(z) + prd_plots.extents(r + 8),
#            cmap='Reds', alpha=0.25)
# plt.imshow(0.1 * S_rz, extent=prd_plots.extents(z) + prd_plots.extents(r - 2),
#            cmap='Reds', alpha=0.25)
# plt.imshow(0.1 * S_rz, extent=prd_plots.extents(z) + prd_plots.extents(r - 5),
#            cmap='Reds', alpha=0.25)
# plt.imshow(0.1 * S_rz, extent=prd_plots.extents(z) + prd_plots.extents(r),
#            cmap='Reds', alpha=0.25)
# plt.imshow(S_rz, extent=prd_plots.extents(z) + prd_plots.extents(r),
#            cmap='Reds')
plt.imshow(I_rz, extent=prd_plots.extents(z) + prd_plots.extents(r),
           cmap='Greens')
plt.plot(z, w_z0, '-', c='xkcd:green', lw=1)
plt.plot(z, -w_z0, '-', c='xkcd:green', lw=1)
# plt.axis('off')
# plt.plot(z, w_z1, c='xkcd:green')
# plt.plot(z, -w_z1, c='xkcd:green')

###### xy plot ###############################################################
# size = 4
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('x axis')
# ax2.set_ylabel('y axis')
# plt.plot(x1, y1, '.', alpha=0.4, color=cs['gglred'], label='')
# plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
# plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

###### xyz plot ##############################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# ax3.set_zlabel('z axis')
# # scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'],
# # label='')
# plt.plot(z, w_z, c=cs['ggred'])
# plt.plot(z, -w_z, c=cs['ggred'])
# # surf = ax3.plot_surface(*coords, I_rz/5, cmap='magma', alpha=0.3)
# contour = ax3.contour(*coords, S_rz / 10, 20, cmap='viridis')


# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# ax3.set_zlim(0, z_lim)

# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.tight_layout()
plot_file_name = plot_path + 'plot11'
plt.savefig(plot_file_name + '.svg')
plt.show()


prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
