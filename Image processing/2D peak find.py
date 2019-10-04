##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from mpl_toolkits.axes_grid1 import make_axes_locatable

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
prd_plots.ggplot()
cs = prd_plots.palette()

##############################################################################
# File paths
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Confocal measurements\SCM Data 20190906"
      r"\Raster scans\06Sep19 scan-011.txt")

p1 = (r"D:\Experimental Data\F5 L10 Confocal measurements\SCM Data 20190907"
      r"\Raster scans\07Sep19 scan-001.txt")

##############################################################################
# Plot image to find point spread function
##############################################################################
fig1 = plt.figure('fig1', figsize=(4, 4))
prd_plots.ggplot()
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x dimension (V)')
ax1.set_ylabel('y dimension (V)')
plt.title(lb)
im1 = plt.imshow(Gk, cmap='magma')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig1.colorbar(im1, cax=cax)
plt.title('select 3 corners of a square surrounding clean PSF')
plt.ginput(3)

x, y, img = prd_file_import.load_SCM_F5L10(p0)
print(np.shape(img))

krnl_size = 25
x_k = np.arange(krnl_size)
y_k = np.arange(krnl_size)

coords = np.meshgrid(x_k, y_k)

G = prd_maths.Gaussian_2D(coords, 1, int(krnl_size - 1) / 2,
                          int(krnl_size - 1) / 2, 2, 2)
G = np.reshape(G, (krnl_size, krnl_size))
print(np.shape(G))
Gk = ndi.convolve(img, G)
lb = 'p0'
##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"

###### plots ###############################################################

fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x dimension (V)')
ax1.set_ylabel('y dimension (V)')
plt.title(lb)
im1 = plt.imshow(Gk, cmap='magma')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig1.colorbar(im1, cax=cax)

fig2 = plt.figure('fig2', figsize=(4, 4))
prd_plots.ggplot()
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x dimension (V)')
ax2.set_ylabel('y dimension (V)')
plt.title(lb)
im2 = plt.imshow(np.flipud(img), cmap='magma',
                 extent=prd_plots.extents(y) +
                 prd_plots.extents(x),
                 label=lb,
                 vmin=np.min(img),
                 vmax=0.6 * np.max(img)
                 )
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig2.colorbar(im2, cax=cax)

plt.show()
