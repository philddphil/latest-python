##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# Specify results directory and change working directory to this location

p0 = (r"C:\local files\Experimental Data\G4 L12 Rennishaw\20191101\spec")
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*.txt')

# Initialise lists of datasets
sz = 5
prd_plots.ggplot()
fig1 = plt.figure('fig1', figsize=(sz * np.sqrt(2), sz))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Spectral signal (arb)')

for i0, val0 in enumerate(datafiles[0:]):
    
    data = np.genfromtxt(val0)
    λ = data[:, 0] 
    cts = data[:, 1] 
    ax1.plot(1e9 * λ, cts)
    plt.xlim((550, 775))
    plt.ylim((0, 1250))
    
plt.show()
ax1.figure.savefig('plot_name' + 'dark.svg')
prd_plots.PPT_save_2d(fig1, ax1, 'fig.svg')
