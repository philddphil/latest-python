##############################################################################
# Import some libraries
##############################################################################
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\G4 L12 Rennishaw\20190829\[0 to 5] h [0 to 1] v from TL corner 100micron NPL areal")

pZ = p0 + r"\data Z"
pXY = p0 + r"\data XY"
pXYZ = p0 + r"\data XYZ"

λs = 1e9 * np.genfromtxt(pZ, skip_header=3)[:, 0]
spec = np.genfromtxt(pZ, skip_header=3)[:, 1]
with open(pXY, 'r') as the_file:
    all_data = [line.strip() for line in the_file.readlines()]
    width_data = all_data[1].split('\t')[0]
    width = float(re.findall(r'[0-9]*[.]*[0-9]', width_data)[0])
    height_data = all_data[2].split('\t')[0]
    height = float(re.findall(r'[0-9]*[.]*[0-9]', height_data)[0])
    im = np.loadtxt(all_data[4:])
xs = np.linspace(0, width, np.shape(im)[1])
ys = np.linspace(0, height, np.shape(im)[0])

zs = np.genfromtxt(pXYZ)
zs = zs.reshape((1015, len(ys), len(xs)))
print(np.shape(zs))

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
# plot_path = r"D:\Python\Plots\\"
# plot_path = r"C:\Users\Phil\Documents\GitHub\plots"

##### image plot ############################################################
fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.imshow(im, extent=prd_plots.extents(xs) + prd_plots.extents(ys))

fig3 = plt.figure('fig3', figsize=(5, 5))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
plt.imshow(zs[0, :, :], extent=prd_plots.extents(xs) + prd_plots.extents(ys))


# plot_file_name = plot_path + 'plot1.png'
# prd_plots.PPT_save_3d(fig2, ax2, plot_file_name)

###### xy plot ###############################################################
size = 4
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(λs, spec, '.', alpha=0.4, color=cs['gglblue'], label='')
plt.show()
