##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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
π = np.pi
c = 299792458
λ0 = 660e-9
λΔ = 10e-9
λres = 9

λs = np.linspace(λ0 - 10 * λΔ, λ0 + 10 * λΔ, λres)
frqs = c / λs
ωs = 2 * π * frqs
ts = np.linspace(0, 50 * 1e-15, 1000)
sins = []

for i0, val0 in enumerate(ωs):
    sins.append(np.cos(val0 * ts))

print(sum(sins))
λspec = λΔ / ((λs - λ0)**2 + λΔ**2)
Wavepacket = np.fft.ifft(λspec)
##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))
x = λs * 1e9
y = λspec / np.max(λspec)

###

fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
for i0, val0 in enumerate(ωs):
    plt.plot(ts, sins[i0] + 2 * i0, '-')
plt.tight_layout()

###

fig2 = plt.figure('fig2', figsize=(5, 5))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(sum(sins), '.:')
plt.tight_layout()

###

fig3 = plt.figure('fig3', figsize=(5, 5))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
plt.plot(frqs, y, '.:')
plt.tight_layout()

###

plt.show()
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
