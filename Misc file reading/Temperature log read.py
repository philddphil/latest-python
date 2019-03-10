##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
from datetime import datetime
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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)\SCM Data 20190128"
      r"\Temperature log.txt")
a = open(p0, 'r', encoding='utf-8')
data = a.readlines()
a.close
t_sec = []
t_date = []
T = []
for i0, val in enumerate(data[0:]):
    t_string = val.split("\t")[0]
    t_datetime = datetime.strptime(t_string, "%d/%m/%Y %H:%M:%S")
    t_sec = np.append(t_sec, t_datetime.timestamp())
    t_date = np.append(t_date, t_datetime)
    T = np.append(T, float(val.split("\t")[1]))

##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
###

fig1 = plt.figure('fig1', figsize=(10, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (K)')
plt.plot(t_date, T, 'o', color=cs['gglred'], alpha=0.2, label='T points')
plt.plot(t_date, T, '-', color=cs['gglblue'], label='T line')
plt.tight_layout()
ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
plt.show()
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
