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
λΔ = 10e-12
λres = 21

λs = np.linspace(λ0 - 15 * λΔ, λ0 + 15 * λΔ, λres)
frqs = c / λs

λspec = λΔ / ((λs - λ0)**2 + λΔ**2)
Wavepacket = np.fft.fft(λspec)
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
plt.plot(Wavepacket, '.:')
plt.tight_layout()

###

fig2 = plt.figure('fig2', figsize=(5, 5))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(frqs, y, '.:')
plt.tight_layout()


plt.show()
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
