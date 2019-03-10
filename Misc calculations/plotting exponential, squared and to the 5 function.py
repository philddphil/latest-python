##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
t = np.linspace(0, 20	, 100)
y1 = np.exp(t)
y2 = t**2
y3 = t**5
##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
fig2 = plt.figure('fig2', figsize=(3, 3))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('~ length of number to factor')
ax2.set_ylabel('~ time (x 10$^6$)')
plt.plot(t, y1/1e6, label='t = e$^x$')
plt.plot(t, y2/1e6, label='t = x$^2$')
plt.plot(t, y3/1e6, label='t = x$^5$')

plt.tight_layout()
ax2.legend(loc='upper left', fancybox=True, framealpha=0.5)
plt.show()
ax2.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
