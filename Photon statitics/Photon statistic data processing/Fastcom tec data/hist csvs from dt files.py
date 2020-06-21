
##############################################################################
# Import some libraries
##############################################################################
import os
import glob
import numpy as np

##############################################################################
# Define functions to be used
##############################################################################


##############################################################################
# Load the data and histogram values
##############################################################################
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211")
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212"
      r"\g4_1MHzPQ_48dB_cont_snippet_3e6")
# d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212\
#   g4_1MHzTxPIC_55dB_cont_snippet_3e6")
d1 = d0 + r'\Py data'
os.chdir(d1)
datafiles = glob.glob(d1 + r'\dts*')
all_hists = []
res = 1000
t_range = 25100
nbins = int(2 * t_range / res + 1)
edges = np.linspace(-t_range, t_range, nbins + 1)

for i0, v0 in enumerate(datafiles[0:]):
    print(v0)
    dts = np.genfromtxt(v0)
    hist, bin_edges = np.histogram(dts, edges)
    all_hists.append(hist)

hists = np.sum(np.asarray(all_hists), axis=0)

np.savetxt("g2_hist.csv", hists, delimiter=",")
np.savetxt("g2_bins.csv", bin_edges, delimiter=",")
