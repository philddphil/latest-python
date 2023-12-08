##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
p_surface = r"C:\Users\Philip\Documents\GitHub\latest-python\library"
p_home = r"C:\Users\Phil\Documents\GitHub\latest-python\library"
p_office = r"D:\Python\Local Repo\library"
sys.path.insert(0, p_office)
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Laser powers\PM100D 20190611")
os.chdir(p0)
datafiles = glob.glob(p0 + r'\*.txt')
datafiles = prd_file_import.natural_sort(datafiles)

##############################################################################
# Plot some figures
##############################################################################

datafiles.reverse()
datafiles_n = len(datafiles)
colors = plt.cm.viridis(np.linspace(0, 1, datafiles_n))

prd_plots.ggplot()
size = 4

fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('time, (s)')
ax1.set_ylabel('Δ Power (μW)')
ax1.set_title('time series')
fig1.tight_layout()

fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Power (μW)')
ax2.set_ylabel('# measurements')
ax2.set_title('histograms')
fig2.tight_layout()

fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('Drive current (mA)')
ax3.set_ylabel('Measured power (μW)')
ax3.set_title('scatter')
fig3.tight_layout()

# accumulated data for post loop import (times, powers, currents)
ts = []
Ps = []
Ids = []

for i0, val0 in enumerate(datafiles[::-1]):
    lb = str(os.path.basename(val0))
    # This bit grabs a value from the file name, when I was saving it
    # according to the drive curred (Id) set on the laser control box
    regex = re.compile(r'\d+')
    Id = float(regex.findall(lb)[0])
    # import datasets
    x, y = prd_file_import.load_PM100_log(val0)

    # scale ys and offset xs
    t = x - np.min(x)
    P = y * 1e6
    lb = os.path.basename(val0)
    gx, gy = prd_maths.Gauss_hist(P)

    # fig1
    ax1.plot(t, P, label=lb, c=colors[i0])
    fig1.tight_layout()

    # fig2
    ax2.plot(gx, gy, c=colors[i0], lw=0.5)
    ax2.hist(P, bins=10, label=lb, alpha=0.5,
             edgecolor=cs['mnk_dgrey'], facecolor=colors[i0])
    fig2.tight_layout()

    # fig3
    ax3.plot(Id * np.ones(P.shape), P, 'o',
             mfc='None',
             mec=colors[i0])
    fig3.tight_layout()
    Ps.append(np.mean(P))
    ts.append(np.mean(t))
    Ids.append(Id)

# save some of the series data
data_name = 'Power series.dat'
data = np.array(Ps)
header = "Powers"
np.savetxt(data_name, da, header=header)

# Perform fit on series data
x_fit = np.linspace(Ids[0], Ids[-1], 1000)
m = Ps[-1] / Ids[-1]
popt, pcov = curve_fit(prd_maths.straight_line,
                       Ids, Ps, p0=[m, 0])
m_str = str(np.round(popt[0], 2))
c_str = str(np.round(popt[1], 2))
fit_lb = 'y = ' + m_str + 'x ' + c_str

# plot series data
ax3.plot(x_fit, prd_maths.straight_line(x_fit, *popt),
         c=cs['ggred'],
         label=fit_lb)
ax3.legend(loc='upper left', fancybox=True, framealpha=1)
plt.show()

# save series plots
ax3.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, 'time series.png')
prd_plots.PPT_save_2d(fig2, ax2, 'histogram.png')
prd_plots.PPT_save_2d(fig3, ax3, 'scatter.png')
