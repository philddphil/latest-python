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
p_office = r"D:\Python\Local Repo\library"
sys.path.insert(0, p_office)
np.set_printoptions(suppress=True)
import prd_plots
import prd_maths
import prd_file_import
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
x = np.linspace(0, 20, 6)
x = x + 2 * np.random.random(x.shape)
m = 1
c = 1
y = m * x + c + 2 * np.random.random(x.shape)
SNR = 0.3
yerr = SNR + np.random.random(x.shape)
xerr = np.random.random(x.shape) + SNR + 0.3
##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
# plot_path = r"D:\Python\Plots\\"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
# plot_path = r"C:\Users\Philip\Documents\GitHub\plots"
plot_path = r"D:\Python\Plots"

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
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('10ET reading (mV)')
ax2.set_ylabel('DUT reading (kct/s)')
ax2.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o',
             elinewidth=0.5, ecolor=cs['gglred'],
             alpha=1, color=cs['ggdred'], label='')
# ax2.grid(False)

plt.tight_layout()


plt.show()
plot_file_name = plot_path + r'\plot1.png'
prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
