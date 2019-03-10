##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)

import prd_plots
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
V0 = 100
BO_rate = 0.009
days = 50
Vs = []
Vt = V0
for i0, j0 in enumerate(np.arange(days)):
    Vt = Vt - 0.009 * Vt
    Vs.append(Vt)
    print('day ', i0, ', V = ', np.round(Vt, 2), 'l')

##############################################################################
# Plot some figures
##############################################################################

prd_plots.ggplot()
###
scale = 5
fig1 = plt.figure('fig1', figsize=(scale * np.sqrt(2), scale * 1))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Time, days')
ax1.set_ylabel('Volume (l)')
plt.plot(np.arange(days), Vs, 'o:', label='He lost forever')
plt.tight_layout()
ax1.legend(loc='upper right', fancybox=True, framealpha=0.5)
ax1.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
plt.show()

prd_plots.PPT_save_2d(fig1, ax1, 'plot1.png')
