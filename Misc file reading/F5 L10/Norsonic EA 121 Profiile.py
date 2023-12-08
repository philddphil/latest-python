
##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
# NPL laptop library path
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
# Surface Pro 4 library path
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
dirpath = r"C:\local files\Experimental Data\G5 A5 Norsonic\200131 0"

filepath1 = r"\NPL_INBOX_0009_Ch1-GLOBAL_LfFspl.txt"
filepath2 = r"\NPL_INBOX_0009_Ch1-RPT1_LfFspl.txt"
filepath3 = r"\NPL_INBOX_0009_Ch1-PROFILE_LfFspl.txt"

# read in the global and report files
hdr1, fs, dBs_f = prd_file_import.load_NEA_Glob(dirpath + filepath1)
hdr2, ts, dBs_tf_R = prd_file_import.load_NEA_Prof(dirpath + filepath2)
print(hdr2)
thrs = [hdr2[0] * v0 / (60 * 60) for v0 in ts]

# the profile data is 50 times more resolved than the report data.
# it can be accessed here for subsequent analysis if needed.
# it is impracticle to plot an image with 800,000 pixels in x, so it is not -
# - accessed here.
# hdr_P, ts_P, dBs_tf_P = prd_file_import.load_NEA_Prof(dirpath + filepath3)

dBs_t = np.sum(dBs_tf_R, 1) / 53
print(np.shape(dBs_f))
print(np.shape(dBs_t))

##############################################################################
# Plot some figures
##############################################################################
# prep colour scheme for plots and paths to save figs to
prd_plots.ggplot()
# NPL path
plot_path = r"C:\local files\Python\Plots"
# Surface Pro path
plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
# os.chdir(plot_path)

# freq bar plot ##############################################################
w = 4
ws = [v0 / w for v0 in fs]
bar_cs = plt.cm.magma(dBs_f / np.max(dBs_f))
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2) / 2, size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_ylabel('frequency, Hz')
ax1.set_yscale('log', basey=2)
ax1.set_xlabel('Avg. dB over time')
ax1.get_yaxis().set_visible(False)
plt.title(' ')
plt.barh(fs, dBs_f, ws, edgecolor=cs['mnk_dgrey'], label='integrated spectrum',
         color=bar_cs)
fig1.tight_layout()

# avg power time plot ########################################################
w = 4
ws = [v0 / w for v0 in fs]
time_cs = plt.cm.magma(dBs_t[0:100] / np.max(dBs_t[0:100]))
size = 4
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size / 2))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('time, hrs')
ax2.set_ylabel('Avg. dB over freq.')
plt.plot(thrs, dBs_t, lw=0.1, color=cs['ggdred'])
plt.plot(thrs, dBs_t, '.', alpha=0.1)

plt.title('t = ' + str(hdr1[0]) +
          's  ' + '# = ' + str(hdr1[1]) +
          ' ' + 'measured @' + str(hdr1[2]))
ax2.get_xaxis().set_visible(False)
fig2.tight_layout()

# img plot ###################################################################
t_ax = thrs
fq_ax = np.flip(fs)
size = 4
fig4 = plt.figure('fig4', figsize=(size * np.sqrt(2), size))
ax4 = fig4.add_subplot(1, 1, 1)
fig4.patch.set_facecolor(cs['mnk_dgrey'])
ax4.set_xlabel('time, hrs')
ax4.set_ylabel('frequency, Hz')
im4 = plt.imshow(np.flipud(np.transpose(dBs_tf_R)),
                 cmap='magma',
                 aspect='auto',
                 extent=[np.min(t_ax), np.max(t_ax),
                         np.min(fq_ax), (np.max(fq_ax))])
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb4 = fig4.colorbar(im4, cax=cax)
plt.show()

# save plots #################################################################
os.chdir(dirpath)

prd_plots.PPT_save_2d(fig1, ax1, 'integrated spectrum')
prd_plots.PPT_save_2d(fig2, ax2, 'time series')
prd_plots.PPT_save_2d_im(fig4, ax4, cb4, 'spectral timeline')
