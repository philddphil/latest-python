##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Confocal measurements\SCM Data 20190403"
      r"\Temperature log.txt")

T, t_date = prd_file_import.load_T_log(p0)
##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"
plot_label = p0.split('\\')[-2]
plot_label = plot_label.split(' ')[-1]
print(plot_label)
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
plot_file_name = plot_path + plot_label + '.png'
prd_plots.PPT_save_2d(fig1, ax1, plot_file_name)
