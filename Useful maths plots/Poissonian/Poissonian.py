##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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
n_max = 15
res = n_max + 1

n_bar1 = 0.1
n_bar2 = 3

n_ints = np.linspace(0, n_max, res)
n_cont = np.linspace(0, n_max, 1024)

P_ints1 = prd_maths.Poissonian_1D(n_ints, n_bar1)
P_cont1 = prd_maths.Poissonian_1D(n_cont, n_bar1)

P_ints2 = prd_maths.Poissonian_1D(n_ints, n_bar2)
P_cont2 = prd_maths.Poissonian_1D(n_cont, n_bar2)


##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))
x_d = n_ints
y_d1 = P_ints1
y_d2 = P_ints2

x_c = n_cont
y_c1 = P_cont1
y_c2 = P_cont2

size = 3
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel(r'$\langle n \rangle$')
ax2.set_ylabel('Probability')
# plt.plot(x1, y1, '.', alpha=0.8, color=cs['gglred'], label=r'$\mathbb{N}$')
plt.bar(x_d, y_d1, alpha=0.5, color=cs['gglred'], label=r'$\mathbb{N}_1$')
plt.bar(x_d, y_d2, alpha=0.5, color=cs['gglpurple'], label=r'$\mathbb{N}_2$')
# plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
plt.plot(x_c, y_c1, alpha=1, color=cs['ggblue'], lw=0.5, label=r'$\mathbb{R}$')
plt.plot(x_c, y_c2, alpha=1, color=cs['ggblue'], lw=0.5)

ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
# os.chdir(p0)
plt.tight_layout()
plt.show()
plot_file_name = plot_path + 'blue nbar = ' + str(n_bar1) + '.png'
ax2.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
