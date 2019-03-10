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
γs = 1e4
Δt = 1


τs_1 = []
for i0, j0 in enumerate(np.arange(γs)):
    t = 0
    emission = 0
    while emission == 0:
        t = t + Δt
        if np.random.random() < 0.01:
            emission = 1
            τs_1.append(t)

τs_2 = []
for i0, j0 in enumerate(np.arange(γs)):
    t = 0
    emission = 0
    while emission == 0:
        t = t + Δt
        if np.random.random() < 0.02:
            emission = 1
            τs_2.append(t)
            print(i0, t)

#######################################################################
# Plot some figures
#######################################################################

prd_plots.ggplot()
###

fig1 = plt.figure('fig1', figsize=(2.5, 2.5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Time')
ax1.set_ylabel('#')
plt.hist(τs_1, bins=20, alpha=0.5,
         facecolor=cs['ggred'], edgecolor=cs['mnk_dgrey'], label='τ = 2')
plt.hist(τs_2, bins=15, alpha=0.5,
         facecolor=cs['ggblue'], edgecolor=cs['mnk_dgrey'], label='τ = 2')
plt.tight_layout()
ax1.legend(loc='upper right', fancybox=True, framealpha=0.5)
plt.savefig('plot.svg')
fig2 = plt.figure('fig2', figsize=(2.5, 2.5))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Time')
ax2.set_ylabel('#')
ax2.set_yscale('log')
plt.hist(τs_1, bins=20, alpha=0.5,
         facecolor=cs['ggred'], edgecolor=cs['mnk_dgrey'], label='τ = 1')
plt.hist(τs_2, bins=15, alpha=0.5,
         facecolor=cs['ggblue'], edgecolor=cs['mnk_dgrey'], label='τ = 2')
plt.tight_layout()
ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
plt.show()
plt.savefig('logplot.svg')
ax2.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
ax1.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))

prd_plots.PPT_save_2d(fig1, ax1, 'plot.png')
prd_plots.PPT_save_2d(fig2, ax2, 'logplot.png')
