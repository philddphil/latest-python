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

# p0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Conferences\SPW 2019\Data\Spectra\hBN spectra"))
p0 = (r"C:\Users\pd10\OneDrive - National Physical Laboratory\Conferences\SPW 2019\Data\Spectra\Abstract figure")
os.chdir(p0)
# Generate list of relevant data files and sort them chronologically
datafiles = glob.glob(p0 + r'\*.txt')
print(datafiles)
# Initialise lists of datasets

prd_plots.ggplot()
fig1 = plt.figure('fig1', figsize=(3 * np.sqrt(2), 3))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Wavelength (λ) / nm')
ax1.set_ylabel('Spectral signal (arb)')
cs_list = (cs['ggred'],cs['ggyellow']
    , cs['mnk_green'], cs['mnk_pink'])
for i0, val0 in enumerate(datafiles[0:]):
    
    data = np.genfromtxt(val0)
    print(data)
    λ = data[:, 0] 
    cts = data[:, 1] 
    ax1.plot(λ, cts/np.max(cts), color=cs_list[i0])
    plt.xlim((550, 950))
    
plt.show()
ax1.figure.savefig('plot_name' + 'dark.svg')
