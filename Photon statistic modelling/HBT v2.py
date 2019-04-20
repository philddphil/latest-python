##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
# Office PC (eea) path
sys.path.insert(0, r"D:\Python\Local Repo\library")

# Surface Pro path
sys.path.insert(0, r"C:\Users\Philip\Documents\GitHub\latest-python\library")

np.set_printoptions(suppress=True)

import prd_plots
import prd_data_proc
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
# load photon timing files generated by 'Generate photons.py'
T = 10
τ_decay = 10
τ_excite = 1
k_12 = 0.01

# Office PC (eea) path
p0 = r'D:\Experimental Data\Python simulations (G5 A5)'\
    r'\Single photon statistics\Data\pulsed ' + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    str(k_12) + 'k_12,' + \
    'T ' + str(round(T, 2)) + '%'
# Surface Pro path
p0 = r"C:\Users\Philip\Documents\Data\pulsed " + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    str(k_12) + 'k_12,' + \
    'T ' + str(round(T, 2)) + '%'

f4 = p0 + ' - HBT.txt'
f5 = p0 + ' - fom.txt'

τs_HBT = np.loadtxt(f4)
with open(f5, 'r', encoding='utf-8') as f:
    a = f.read()
    b = a.split('\n')
    for i0, j0 in enumerate(b):
        if 'exp clk =' in j0:
            t_clk = float(j0.split(' = ')[-1])
        if 'τ_excite =' in j0:
            τ_excite = float(j0.split(' = ')[-1])
        if 'τ_decay =' in j0:
            τ = float(j0.split(' = ')[-1])
        if 'Δt =' in j0:
            Δt = float(j0.split(' = ')[-1])
        if '# photons =' in j0:
            γs = float(j0.split(' = ')[-1])

print(len(τs_HBT))
print('experiment time = ', 1e-6 * t_clk, 'ms')
print('total count rate = ', 1e3 * (γs / t_clk), 'Mcps')
# Prepare histogram to be plotted
x1 = τs_HBT
x1 = [i for i in x1 if i <= 500]
x1 = [i for i in x1 if i >= -500]
print('plotted data', len(x1))
print('norm factor =', ((γs)**2 / (4 * t_clk)))

bin_N = 201
h1, bins = np.histogram(x1, bins=bin_N)
bin_centres = np.linspace(bins[0] + (bins[1] - bins[0]) / 2,
                          bins[-2] + (bins[-1] - bins[-2]) / 2, len(bins) - 1)
hist_time = np.linspace(bins[0] + (bins[1] - bins[0]) / 2,
                        bins[-2] + (bins[-1] - bins[-2]) / 2, 1000)
bin_width = bins[1] - bins[0]
print(bin_width)
print(bin_width * γs)
# Normailise τs_HBT to g2 values
det_rate = 0.5 * (γs / t_clk)
g2s = h1 / (det_rate * det_rate * bin_width * t_clk)
x2 = 100 * g2s

#######################################################################
# Plot some figures
#######################################################################
prd_plots.ggplot()

# histograms
fig2 = plt.figure('fig2', figsize=(6 * np.sqrt(2), 6))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('τ (ns)')
ax2.set_ylabel('#')
plt.hist(x1, bins=bin_N, alpha=0.5,
         facecolor=cs['ggred'], edgecolor=cs['mnk_dgrey'],
         label='Probabilistic')
plt.plot(bin_centres, h1, '.', color=cs['ggblue'])
plt.savefig('hist.svg')
plt.title('total count rate = ' + str(int(1e3 * (γs / t_clk))) + ' Mcps')
plt.tight_layout()
# plt.ylim(0, 1.5 * np.mean(hist))

fig1 = plt.figure('fig1', figsize=(6 * np.sqrt(2), 6))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('τ (ns)')
ax1.set_ylabel('g$^2$(τ)')
plt.plot(bin_centres, x2, '.', color=cs['ggblue'])
plt.savefig('g2s.svg')
plt.title('total count rate = ' + str(int(1e3 * (γs / t_clk))) + ' Mcps')
plt.tight_layout()
# plt.ylim(0, 1.5 * np.mean(hist))
plt.show()
prd_plots.PPT_save_2d(fig1, ax1, p0 + '-g2.png')
prd_plots.PPT_save_2d(fig2, ax2, p0 + '-hist.png')
