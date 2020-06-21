
##############################################################################
# Import some libraries
##############################################################################
import glob
import numpy as np

##############################################################################
# Define functions to be used
##############################################################################


##############################################################################
# Load the data and histogram values
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 Confocal measurements"
      r"\SCM Data 20200310\HH T3 142814\t range 100000 i2 range 0")
# p0 = (r"C:\Users\pd10\Documents\HH Data to process\20200310 130500")
      # r"\t range 100000 i2 range 0")

datafiles = glob.glob(p0 + r'\dts*')
all_hists = []
res = 0.4
t_range = 100000
nbins = int(2 * t_range / res)
print(nbins)

for i0, v0 in enumerate(datafiles[0:]):
    print(i0)
    dts = np.genfromtxt(v0)
    hist, bin_edges = np.histogram(dts, nbins, (-t_range, t_range))
    all_hists.append(hist)

hists = np.sum(np.asarray(all_hists), axis=0)
print(np.max(hist))
np.savetxt("400ps_res_hist.csv", hists, delimiter=",")
np.savetxt("400ps_res_bins.csv", bin_edges, delimiter=",")
