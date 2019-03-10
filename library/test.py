##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20190222"
      r"\Raster scans")
datafiles = glob.glob(p0 + r'\*.txt')
filepath = datafiles[0]
a = open(filepath, 'r', encoding='utf-8')
data = a.readlines()
a.close()
for i0, j0 in enumerate(data):
    if 'X initial / V' in j0:
        x_init = float(data[i0].split("\t")[-1])
    if 'X final / V' in j0:
        x_fin = float(data[i0].split("\t")[-1])
    if 'X increment / V' in j0:
        x_res = float(data[i0].split("\t")[-1])
    if 'Y initial / V' in j0:
        y_init = float(data[i0].split("\t")[-1])
        print(y_init)
    if 'Y final / V' in j0:
        y_fin = float(data[i0].split("\t")[-1])
        print(y_fin)
    if 'Y increment / V' in j0:
        y_res = float(data[i0].split("\t")[-1])
    if 'Y wait period / ms' in j0:
        data_start_line = i0 + 2
        print(data_start_line)

x = np.linspace(x_init, x_fin, x_res)
y = np.linspace(y_fin, y_init, y_res)
img = np.loadtxt(data[data_start_line:])
