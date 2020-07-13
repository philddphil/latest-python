
##############################################################################
# Import some libraries
##############################################################################
import os
import glob
import numpy as np

##############################################################################
# Load the data and histogram values
##############################################################################
d0 = (r"C:\local files\Experimental Data"
      r"\F5 L10 Confocal measurements\SCM Data 20200707")
d1 = d0 + r'\HH T3 145149'
d2 = d1 + r'\time difference files'

os.chdir(d2)
datafiles = glob.glob(d2 + r'\dts_chs01_*')

res = 4
t_range = 25100
nbins = int(2 * t_range / res + 1)
edges = np.linspace(-t_range, t_range, nbins + 1)
hists = np.zeros(nbins)

for i0, v0 in enumerate(datafiles[0:]):
    print(v0)
    dts = np.genfromtxt(v0)
    hist, bin_edges = np.histogram(dts, edges)
    hists+=hist

np.savetxt("g2_hist.csv", hists, delimiter=",")
np.savetxt("g2_bins.csv", bin_edges, delimiter=",")
