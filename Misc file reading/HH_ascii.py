##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()
##############################################################################
# Do some stuff
##############################################################################
p0 = r"D:\Experimental Data\F5 L10 HydraHarp\HH data 20190903\tt0.txt"
p1 = r"D:\Experimental Data\F5 L10 HydraHarp\HH data 20190903\tt1.txt"

tt0 = np.loadtxt(p0)
tt1 = np.loadtxt(p1)

δt0 = []
for i0, val0 in enumerate(tt0):
    if i0 == 0:
        δt0.append(0)
    else:
        δt0.append(val0 - tt0[i0 - 1])
δt1 = []
for i0, val0 in enumerate(tt1):
    if i0 == 0:
        δt1.append(0)
    else:
        δt1.append(val0 - tt1[i0 - 1])

with open('deltat0.txt', 'w') as f:
    for item in δt0:
        f.write("%s\n" % item)

with open('deltat1.txt', 'w') as f:
    for item in δt1:
        f.write("%s\n" % item)

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"

###### xy plot ###############################################################
size = 4
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Δt (ps)')
ax2.set_ylabel('freq #')
plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)
ax2.set_yscale('log')

size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Δt (ps)')
ax2.set_ylabel('freq #')
plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)

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
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

plt.show()
ax2.figure.savefig('SPAD histogram' + '.jpg')
# plot_file_name = plot_path + 'plot1.png'
# prd_plots.PPT_save_3d(fig2, ax2, plot_file_name)
