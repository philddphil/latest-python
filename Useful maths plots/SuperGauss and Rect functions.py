# ****************************** HOUSEKEEPING ****************************
from numpy import*
from pylab import *
from math import*
from scipy import signal
import sys
import time
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
cs = prd.palette()

r = 3
x = np.linspace(-r, r)
(X, Y) = np.meshgrid(x, x)
coords = (X, Y)
L = len(x)
print(np.shape(X))
G0 = prd.Gaussian_2D(coords, 1, 0, 0, 1, 1, 0, 0, 1)
G1 = np.reshape(G0, np.shape(X))
G0 = prd.Gaussian_2D(coords, 1, 0, 0, 1, 1, 0, 0, 3)
G2 = np.reshape(G0, np.shape(X))
G0 = prd.Gaussian_2D(coords, 1, 0, 0, 1, 1, 0, 0, 50)
G3 = np.reshape(G0, np.shape(X))

R1 = np.exp(-1 * (X + Y)**2)
R2 = zeros(np.shape(X))
R2[int(L / 4):int(3 * L / 4),int(L / 4):int(3 * L / 4)] = 1
R3 = prd.n_G_blurs(R2,1,2)
R4 = prd.n_G_blurs(R2,1,6)

# fig0 = plt.figure('fig0')
# fig0.patch.set_facecolor(cs['mdk_dgrey'])
# ax0 = Axes3D(fig0)
# gcs1 = cm.viridis(G1)
# gcs2 = cm.viridis(G2)
# gcs3 = cm.viridis(G3)
# rcs2 = cm.viridis(R2)
# rcount, ccount, _ = gcs1.shape
# ax0.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.set_xlabel('x axis')
# ax0.set_ylabel('y axis')
# ax0.set_zlabel('z axis')
# surfg1 = ax0.plot_surface(X, Y, G1, rcount=rcount, ccount=ccount,
#                           facecolors=gcs1, shade=False,
#                           alpha=0.6)
# surfg1.set_facecolor((0, 0, 0, 0))
# surfg2 = ax0.plot_surface(X + 2 * r, Y, G2, rcount=rcount, ccount=ccount,
#                           facecolors=gcs2, shade=False,
#                           alpha=0.6)
# surfg2.set_facecolor((0, 0, 0, 0))
# surfg3 = ax0.plot_surface(X + 4 * r, Y, G3, rcount=rcount, ccount=ccount,
#                           facecolors=gcs3, shade=False,
#                           alpha=0.6)
# surfg3.set_facecolor((0, 0, 0, 0))
# surfr1 = ax0.plot_surface(X + 6 * r, Y, R2, rcount=rcount, ccount=ccount,
#                           facecolors=rcs2, shade=False,
#                           alpha=0.6)
# surfr1.set_facecolor((0, 0, 0, 0))
# ax0.set_aspect(1)

F = np.concatenate((G1,G2,G3,R1,R2,R3))

fig1 = plt.figure('fig1')
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1_1 = fig1.add_subplot(3, 2, 1)
plt.imshow(G1)
ax1_1 = fig1.add_subplot(3, 2, 3)
plt.imshow(G2)
ax1_1 = fig1.add_subplot(3, 2, 5)
plt.imshow(G3)
ax1_1 = fig1.add_subplot(3, 2, 2)
plt.imshow(R2)
ax1_1 = fig1.add_subplot(3, 2, 4)
plt.imshow(R3)
ax1_1 = fig1.add_subplot(3, 2, 6)
plt.imshow(R4)

plt.show()
