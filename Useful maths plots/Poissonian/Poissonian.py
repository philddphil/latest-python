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
n_max = 10
res = n_max + 1
n_bar = 3

n_ints = np.linspace(0, n_max, res)
n_cont = np.linspace(0, n_max, 1024)

P_ints = prd_maths.Poissonian(n_ints, n_bar)
P_cont = prd_maths.Poissonian(n_cont, n_bar)


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
x1 = n_ints
y1 = P_ints

x2 = n_cont
y2 = P_cont

size = 4
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(x1, y1, '.', alpha=0.8, color=cs['gglred'], label=r'$\mathbb{N}$')
# plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label=r'$\mathbb{R}$')

ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
# os.chdir(p0)
plt.tight_layout()
plt.show()
plot_file_name = plot_path + 'plot1.png'
prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
