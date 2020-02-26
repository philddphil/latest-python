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
# Import data (saved by Rob, usisng fastcom tech tt)
##############################################################################
d0 = r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211"
p1 = d0 + r"\1.txt"
p2 = d0 + r"\2.txt"
p3 = d0 + r"\3.txt"
p4 = d0 + r"\4.txt"
p6 = d0 + r"\6.txt"

os.chdir(d0)

# 1e-7 is the saved resolution - this is 0.1 nanosecond
tt1 = np.loadtxt(p1)
tt2 = np.loadtxt(p2)
tt3 = np.loadtxt(p3)
tt4 = np.loadtxt(p4)
tt6 = np.loadtxt(p6)

# convert to ns
tt1_ns = [j0 * 1e1 for j0 in tt1]
tt2_ns = [j0 * 1e1 for j0 in tt2]
tt3_ns = [j0 * 1e1 for j0 in tt3]
tt4_ns = [j0 * 1e1 for j0 in tt4]
tt6_ns = [j0 * 1e1 for j0 in tt6]


# calc total time and # counts, count rates & gradient functions
total_t = np.max(tt6_ns) * 1e-9
# global_t += total_t
# c0 = len(tt0_ns)
# c1 = len(tt1_ns)

# cps0 = c0 / total_t
# cps1 = c1 / total_t

# global_cps0.append(cps0)
# global_cps1.append(cps1)

# dydx0 = np.max(tt0_ns) / len(tt0_ns)
# dydx1 = np.max(tt1_ns) / len(tt1_ns)

# # print some values
# print('total time collected ', np.round(total_t, 5), 's')
# print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
# print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')

# ##########################################################################
# # Calculate dt (time difference series)
# dt0_ns.extend(np.diff(tt0_ns))
# dt1_ns.extend(np.diff(tt1_ns))

#     ##########################################################################
# np.shape(dt1_ns)
# global_cps0 = np.mean(global_cps0)
# global_cps1 = np.mean(global_cps1)

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
plt.plot(tt1_ns, '.')
plt.plot(tt6_ns, '.')
plt.show()

# save figure
prd_plots.PPT_save_2d(fig1, ax1, 'hist_ts')
