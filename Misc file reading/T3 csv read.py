
##############################################################################
# Import some libraries
##############################################################################
import numpy as np
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
# Do some stuff
##############################################################################

filepath = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
            r"\SCM Data 20200128\HH\HH T3 123912\HBT_ts.csv")

HBT = np.genfromtxt(filepath)
print(np.shape(HBT))
cts, bins = np.histogram(HBT, 2000)
bin_w = (bins[1] - bins[0])/2
print(bin_w * 2)
ts = np.linspace(bins[1] + bin_w, bins[-1] - bin_w, len(bins) - 1)

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

# xy plot ####################################################################

# hist/bar plot ##############################################################
size = 5
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(111)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('δt, ns', fontsize=14, labelpad=80,)
ax1.set_ylabel('Counts', fontsize=14)
ax1.set_ylim(0, max(cts))
plt.hist(HBT, 2000)

size = 5
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('δt, ns', fontsize=14, labelpad=80,)
ax2.set_ylabel('Counts', fontsize=14)
ax2.set_ylim(0, max(cts))
plt.plot(ts, cts)

plt.show()
