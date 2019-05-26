##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
p_surface = r"C:\Users\Philip\Documents\GitHub\latest-python\library"
p_home = r"C:\Users\Phil\Documents\GitHub\latest-python\library"
sys.path.insert(0, p_home)
np.set_printoptions(suppress=True)
import prd_plots
import prd_maths
import prd_file_import
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
x = np.linspace(0.9, 1.1, 500)
A = 50
SNR = 1 / 20
bins = 15
value = 10
noise = np.random.normal(0, A * SNR, x.shape) + value
Gauss_x, Gauss_y = prd_maths.Gauss_hist(noise, bins)
##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
# plot_path = r"D:\Python\Plots\\"
plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
# plot_path = r"C:\Users\Philip\Documents\GitHub\plots"
###### image plot ############################################################
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

###### xy plot ###############################################################
size = 4

fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = fig2.add_subplot(gs[0])
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('measurement number')
ax2.set_ylabel('value')
plt.plot(noise, '.', alpha=1, color=cs['gglred'], label='')
plt.plot(noise, alpha=0.2, color=cs['gglred'], label='')
[ymin, ymax] = ax2.get_ylim()

ax3 = fig2.add_subplot(gs[1])
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('freq of occurances (#)')
ax3.set_ylabel('value')
plt.hist(noise, bins, edgecolor=cs['mnk_dgrey'], color=cs['ggred'], alpha=0.5,
         orientation="horizontal")
plt.plot(Gauss_y, Gauss_x, alpha=1, color=cs['gglblue'], label='')
ax3.set_ylim(ymin, ymax)
# plt.hist(G_noise, 10, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
# plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

###### xyz plot ##############################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'], label='')
# surffit = ax3.contour(*coords, z, 10, cmap=cm.jet)

# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

plt.show()
plot_file_name = plot_path + 'plot1.png'
prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
