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
import prd_data_proc
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# load photon timing files generated by 'Generate photons.py'
T = 5
τ_decay = 1
τ_excite = 10

name_str = str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    'T ' + str(round(T, 2)) + '%'
p0 = r'D:\Experimental Data\Python simulations (G5 A5)'\
    r'\Single photon statistics\Data'
f4 = p0 + r'\\' + name_str + ' - HBT.txt'
f5 = p0 + r'\\' + name_str + ' - fom.txt'

τs_HBT = np.loadtxt(f4)
fom = np.loadtxt(f5)
τ = fom[0]
Δt = fom[1]
γs = fom[2]
print(len(τs_HBT))

# normalise count rates to g2(τ)
g2s = τs_HBT

#######################################################################
# Plot some figures
#######################################################################
prd_plots.ggplot()

# histograms
x1 = τs_HBT
x1 = [i for i in x1 if i <= 100]
x1 = [i for i in x1 if i >= -100]

bin_N = 201
hist, bins = np.histogram(x1, bins=bin_N)
hist_time = np.linspace(bins[0] + (bins[1] - bins[0]) / 2,
                        bins[-2] + (bins[-1] - bins[-2]) / 2, 1000)

fig2 = plt.figure('fig2', figsize=(3 * np.sqrt(2), 3))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Time')
ax2.set_ylabel('#')
plt.hist(x1, bins=bin_N, alpha=0.5,
         facecolor=cs['ggred'], edgecolor=cs['mnk_dgrey'],
         label='Probabilistic')
plt.savefig('logplot.svg')
plt.title('τs_HBT')
plt.tight_layout()
# plt.ylim(0, 1.5 * np.mean(hist))
plt.show()

prd_plots.PPT_save_2d(fig2, ax2, f4 + '.png')
