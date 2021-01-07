##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import glob
import time
import re
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import socket
import scipy as sp
import scipy.io as io
import importlib.util
import ntpath

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from matplotlib import cm

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
# Gaussian beam propagating in the +- x direction, y is the radial ############
def Gaussian_beam(z, r, w0, λ, I0=1):
	# see "Gaussian beam" wikipedia
	k = 2 * np.pi / λ
	zR = np.pi * w0**2 / λ
	w_z = w0 * np.sqrt(1 + (z / zR)**2)
	R_z = z * (1 + (zR / z)**2)
	I_rz = I0 * (w0 / w_z)**2 * np.exp((-2 * r**2) / (w_z**2))
	return I_rz

# Custom palette for plotting ################################################
def palette():
	colours = {'mnk_purple': [145 / 255, 125 / 255, 240 / 255],
			   'mnk_dgrey': [39 / 255, 40 / 255, 34 / 255],
			   'mnk_lgrey': [96 / 255, 96 / 255, 84 / 255],
			   'mnk_green': [95 / 255, 164 / 255, 44 / 255],
			   'mnk_yellow': [229 / 255, 220 / 255, 90 / 255],
			   'mnk_blue': [75 / 255, 179 / 255, 232 / 255],
			   'mnk_orange': [224 / 255, 134 / 255, 31 / 255],
			   'mnk_pink': [180 / 255, 38 / 255, 86 / 255],
			   ####
			   'rmp_dblue': [12 / 255, 35 / 255, 218 / 255],
			   'rmp_lblue': [46 / 255, 38 / 255, 86 / 255],
			   'rmp_pink': [210 / 255, 76 / 255, 197 / 255],
			   'rmp_green': [90 / 255, 166 / 255, 60 / 255],
			   ####
			   'fibre9l_1': [234 / 255, 170 / 255, 255 / 255],
			   'fibre9l_2': [255 / 255, 108 / 255, 134 / 255],
			   'fibre9l_3': [255 / 255, 182 / 255, 100 / 255],
			   'fibre9l_4': [180 / 255, 151 / 255, 255 / 255],
			   'fibre9l_6': [248 / 255, 255 / 255, 136 / 255],
			   'fibre9l_7': [136 / 255, 172 / 255, 255 / 255],
			   'fibre9l_8': [133 / 255, 255 / 255, 226 / 255],
			   'fibre9l_9': [135 / 255, 255 / 255, 132 / 255],
			   'fibre9d_1': [95 / 255, 0 / 255, 125 / 255],
			   'fibre9d_2': [157 / 255, 0 / 255, 28 / 255],
			   'fibre9d_3': [155 / 255, 82 / 255, 0 / 255],
			   'fibre9d_4': [40 / 255, 0 / 255, 147 / 255],
			   'fibre9d_6': [119 / 255, 125 / 255, 0 / 255],
			   'fibre9d_7': [0 / 255, 39 / 255, 139 / 255],
			   'fibre9d_8': [0 / 255, 106 / 255, 85 / 255],
			   'fibre9d_9': [53 / 255, 119 / 255, 0 / 255],
			   ####
			   'ggred': [217 / 255, 83 / 255, 25 / 255],
			   'ggblue': [30 / 255, 144 / 255, 229 / 255],
			   'ggpurple': [145 / 255, 125 / 255, 240 / 255],
			   'ggyellow': [229 / 255, 220 / 255, 90 / 255],
			   'gggrey': [118 / 255, 118 / 255, 118 / 255],
			   'gglred': [237 / 255, 103 / 255, 55 / 255],
			   'gglblue': [20 / 255, 134 / 255, 209 / 255],
			   'gglpurple': [165 / 255, 145 / 255, 255 / 255],
			   'gglyellow': [249 / 255, 240 / 255, 110 / 255],
			   'ggdred': [197 / 255, 63 / 255, 5 / 255],
			   'ggdblue': [0 / 255, 94 / 255, 169 / 255],
			   'ggdpurple': [125 / 255, 105 / 255, 220 / 255],
			   'ggdyellow': [209 / 255, 200 / 255, 70 / 255],
			   }
	return colours


# set rcParams for nice plots #################################################
def ggplot():
	colours = palette()
	plt.style.use('ggplot')
	plt.rcParams['font.size'] = 8
	plt.rcParams['font.family'] = 'monospace'
	plt.rcParams['font.fantasy'] = 'Nimbus Mono'
	plt.rcParams['axes.labelsize'] = 8
	plt.rcParams['axes.labelweight'] = 'normal'
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['legend.fontsize'] = 10
	plt.rcParams['figure.titlesize'] = 8
	plt.rcParams['lines.color'] = 'white'
	plt.rcParams['text.color'] = colours['mnk_purple']
	plt.rcParams['axes.labelcolor'] = colours['mnk_yellow']
	plt.rcParams['xtick.color'] = colours['mnk_purple']
	plt.rcParams['ytick.color'] = colours['mnk_purple']
	plt.rcParams['axes.edgecolor'] = colours['mnk_lgrey']
	plt.rcParams['savefig.edgecolor'] = colours['mnk_lgrey']
	plt.rcParams['axes.facecolor'] = colours['mnk_dgrey']
	plt.rcParams['savefig.facecolor'] = colours['mnk_dgrey']
	plt.rcParams['grid.color'] = colours['mnk_lgrey']
	plt.rcParams['grid.linestyle'] = ':'
	plt.rcParams['axes.titlepad'] = 6


# Set up figure for plotting #################################################
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=4):
	ggplot()
	cs = palette()
	fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
	ax1 = fig1.add_subplot(111)
	fig1.patch.set_facecolor(cs['mnk_dgrey'])
	ax1.set_xlabel(xaxis)
	ax1.set_ylabel(yaxis)
	return ax1, fig1, cs


# For use with extents in imshow ##############################################
def extents(f):
	delta = f[1] - f[0]
	return [f[0] - delta / 2, f[-1] + delta / 2]


##############################################################################
# Do some stuff
##############################################################################
p0 = r"path line 1"\
	r"path line 2"
w0 = 1
λ = 0.7
r = np.linspace(-2, 2, 100)
z = np.linspace(-20, 20, 100)
coords = np.meshgrid(z, r)
I_rz = Gaussian_beam(*coords, w0, λ)
print(np.shape(I_rz))
z_lim = 1
for i0, j0 in enumerate(I_rz):
	for i1, j1 in enumerate(j0):
		print(j1)
		if j1 >= z_lim:
			I_rz[i0, i1] = z_lim

##############################################################################
# Plot some figures
##############################################################################


###### image plot ############################################################
ax1, fig1, cs =  set_figure('Gaussian beam', 'z axis', 'r axis')

plt.imshow(z_lim - I_rz, extent=extents(z) + extents(r),
		   cmap='Greens')

###### xy plot ###############################################################
# size = 4
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('x axis')
# ax2.set_ylabel('y axis')
# plt.plot(x1, y1, '.', alpha=0.4, color=cs['gglred'], label='')
# plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
# plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

###### xyz plot ##############################################################

ax3, fig3, cs = set_figure('Gaussian beam 3D', 'z axis', 'r axis')
ax3 = fig3.add_subplot(111, projection='3d')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'],
# label='')
contour = ax3.contour(*coords, I_rz, 10, cmap=cm.jet)
surface = ax3.plot_surface(*coords, I_rz, cmap=cm.jet, alpha=0.1)

ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
ax3.set_zlim(0, z_lim)

plt.tight_layout()
ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.show()
plot_file_name = plot_path + 'plot1.png'
prd_plots.PPT_save_3d(fig3, ax3, plot_file_name)
