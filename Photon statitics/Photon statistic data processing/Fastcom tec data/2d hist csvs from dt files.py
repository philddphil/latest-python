
##############################################################################
# Import some libraries
##############################################################################
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

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
datafiles = glob.glob(d1 + r'\2d_dts*')
res = 1000
t_range = 25100

nbins = int(2 * t_range / res) + 1
x_edges = np.linspace(-t_range, t_range, nbins + 1)
y_edges = np.linspace(-t_range, t_range, nbins + 1)
dts = []

data = list(csv.reader(open(datafiles[0])))
dt1s = []
dt2s = []
hists = np.zeros((nbins, nbins))

hist = []

for i0, v0 in enumerate(datafiles[0:]):
    data = list(csv.reader(open(datafiles[i0])))
    dt1s = []
    dt2s = []
    for i0, v0 in enumerate(data):
        if len(v0) == 2:
            dt1s.append(int(float(v0[0])))
            dt2s.append(int(float(v0[1])))
    dt1s_a = np.asarray(dt1s)
    dt2s_a = np.asarray(dt2s)
    # dt1s_int = dt1s_a.astype(np.int32)
    # dt2s_int = dt2s_a.astype(np.int32)
    hist, _, _ = np.histogram2d(dt1s_a, dt2s_a, [x_edges, y_edges])
    hists = hists + hist

np.savetxt("g3_hist.csv", hists, delimiter=",")
np.savetxt("g3_xbins.csv", x_edges, delimiter=",")
np.savetxt("g3_ybins.csv", y_edges, delimiter=",")
