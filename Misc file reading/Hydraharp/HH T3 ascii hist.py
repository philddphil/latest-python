##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

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
# Import data (saved by labVIEW code controlling HH400)
##############################################################################
d0 = r"C:\local files\Experimental Data\F5 L10 Confocal measurements" + \
    r"\SCM Data 20200128\HH\HH T3 123912"
p0 = d0 + r"\0 tt ch0.txt"
p1 = d0 + r"\1 tt ch1.txt"
os.chdir(d0)
datafiles0 = glob.glob(d0 + r'\*0.txt')
datafiles1 = glob.glob(d0 + r'\*1.txt')

global_cps0 = []
global_cps1 = []
dt0_ns = []
dt1_ns = []
global_t = 0

for i0, v0 in enumerate(datafiles0[0:]):

    # i0 = 2
    print('file #', i0, ' ', datafiles0[i0])
    # 1e-7 is the saved resolution - this is 0.1 microsecond
    tta = np.loadtxt(datafiles0[i0])
    ttb = np.loadtxt(datafiles1[i0])

    # convert to ns
    tt0_ns = [j0 * 1e2 for j0 in tta]
    tt1_ns = [j0 * 1e2 for j0 in ttb]

    # calc total time and # counts, count rates & gradient functions
    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns)]) * 1e-9
    global_t += total_t
    c0 = len(tt0_ns)
    c1 = len(tt1_ns)

    cps0 = c0 / total_t
    cps1 = c1 / total_t

    global_cps0.append(cps0)
    global_cps1.append(cps1)

    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)

    # print some values
    print('total time collected ', np.round(total_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')

    ##########################################################################
    # Calculate dt (time difference series)
    dt0_ns.extend(np.diff(tt0_ns))
    dt1_ns.extend(np.diff(tt1_ns))

    ##########################################################################
np.shape(dt1_ns)
global_cps0 = np.mean(global_cps0)
global_cps1 = np.mean(global_cps1)

##############################################################################
# Plot some figures
##############################################################################
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"
prd_plots.ggplot()
# xy plot ####################################################################
size = 4
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.hist(dt0_ns, bins=200, range=(0, 150),
         edgecolor=cs['mnk_dgrey'], alpha=0.8)
plt.hist(dt1_ns, bins=200, range=(0, 150),
         edgecolor=cs['mnk_dgrey'], alpha=0.8)
fig1.tight_layout()
plt.show()

# save figure
prd_plots.PPT_save_2d(fig1, ax1, 'hist_ts')
