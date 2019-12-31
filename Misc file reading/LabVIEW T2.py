##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Import and process photon arrival times
##############################################################################
# Set data path and import files as Channel arrival times
p0 = r"C:\local files\Experimental Data\F5 L10 HydraHarp\HH data 20191210\NV"
ch1 = np.genfromtxt(p0 + r'\tt0.txt')
ch2 = np.genfromtxt(p0 + r'\tt1.txt')
print(np.shape(ch1)[0])
print(np.shape(ch2)[0])
total_counts = np.shape(ch1)[0] + np.shape(ch2)[0]
total_time = 1e-7 * np.max([np.max(ch1), np.max(ch2)])

# Generate time series based on photon arrival times
ts1 = np.zeros(int(np.max(ch1)))
ts2 = np.zeros(int(np.max(ch2)))
for i0, v0 in enumerate(ch1):
    ts1[int(v0) - 1] = 1

for i0, v0 in enumerate(ch2):
    ts2[int(v0) - 1] = 1

# Set windowing time for correlation
t_win = 1000

# Trim time series to be multiple of t_win
start_time = time.time()
# g = np.correlate(ts1[0:500000], ts2[0:500000], 'full')
calc_time = time.time() - start_time
print('Calculation time = ', calc_time)

print('total time = ', total_time)
print('total counts = ', total_counts)
print('photon rate (cps) = ', np.round(total_counts / total_time))

##############################################################################
# Import 2 time series of photon arrival times (in ps) from HH/LabVIEW .txts
##############################################################################

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"C:\local files\Python\Plots"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
os.chdir(plot_path)

###### xy plot ###############################################################
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('n$^{th}$ photon')
ax1.set_ylabel('time, seconds')
plt.plot(1e-7 * ch1, '-')
# plt.plot(1e-7 * ch2, '-')
# plt.show()

###### hist/bar plot #########################################################
# size = 9
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Country', fontsize=28, labelpad=80,)
# ax2.set_ylabel('Money (M$)', fontsize=28)
# plt.bar(1, 500, color=cs['ggred'])
# plt.bar(2, 1000, color=cs['ggblue'])
# plt.bar(3, 1275, color=cs['mnk_green'])
# plt.bar(4, 10000, color=cs['ggpurple'])
# ax2.set_xlim(0.5, 4.5)
# ax2.set_ylim(0, 11000)
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Δt (ps)')
# ax2.set_ylabel('freq #')
# plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
# plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)

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

###### save plot ##############################################################
plt.show()
ax1.figure.savefig('plot1' + '.png')
plot_file_name = plot_path + 'plot2.png'
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
