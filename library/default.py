##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations


##############################################################################
# Some defs
##############################################################################
# Prepare the directories and channel names ##################################
def prep_dirs(d0):
    d1 = d0 + r'\py data'
    d2 = d1 + r'\time difference files'
    d3 = d1 + r"\arrival time files"

    try:
        os.mkdir(d1)
    except:
        pass
    try:
        os.mkdir(d2)
    except:
        pass
    try:
        os.mkdir(d3)
    except:
        pass

    return d1, d2, d3


##############################################################################
# Do some stuff
##############################################################################
d0 = (r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech"
      r"\20200807\10 Hrs")
d1, d2, d3 = prep_dirs(d0)

datafiles0 = glob.glob(d3 + r'\*' + 'ch0' + r'*')
datafiles1 = glob.glob(d3 + r'\*' + 'ch1' + r'*')
datafiles2 = glob.glob(d3 + r'\*' + 'ch2' + r'*')
datafiles3 = glob.glob(d3 + r'\*' + 'ch3' + r'*')

print(len(datafiles0))
print(len(datafiles1))
print(len(datafiles2))
print(len(datafiles3))

# tts = 0

# for i0, v0 in enumerate(datafiles0):

#     print(i0)

#     TT = np.load(datafiles3[i0], allow_pickle=True)

#     for i1, v0 in enumerate(TT):

#         tt = TT[i1]

#         tts += len(tt)
#         print(tts)

#     TT = []

chA = 'ch3'
chB = 'ch2'

d30 = d2 + r"\dts " + chA + " & " + chB
d31 = d2 + r"\dts " + chB + " & " + chA

os.chdir(d3)
a = re.findall('\d+', chA)[0]
b = re.findall('\d+', chB)[0]

file_str0 = (r'\dts ' + chA + ' ' + chB)
file_str1 = (r'\dts ' + chB + ' ' + chA)

datafiles0 = glob.glob(d30 + file_str0 + r'*')
datafiles1 = glob.glob(d31 + file_str1 + r'*')

print(len(datafiles0), len(datafiles1))
all_dts0, all_dts1, all_dts_both = 0, 0, 0

for i0, v0 in enumerate(datafiles0[0:]):

    dts0 = np.load(v0, allow_pickle=True)
    dts1 = np.load(datafiles1[i0], allow_pickle=True)

    dts01 = list(set(dts0)|set(dts1))
    all_dts0 += len(dts0)
    all_dts1 += len(dts1)
    all_dts_both += len(dts01)

    # print(dts0[0:10])
    # print(dts1[0:10])
    # print(dts01[0:10])
    dts0 = []
    dts1 = []
    dts01 = []
    print(all_dts0, '\t', all_dts1, '\t', all_dts_both, '\t',)


##############################################################################
# Plot some figures
##############################################################################
# os.chdir(r"C:\local files\Python\Plots")
# xy plot ####################################################################

# ax1, fig1, cs = set_figure(name='figure',
#                            xaxis='x',
#                            yaxis='y',
#                            size=4)
# ax1.plot(lens_ch3)
# ax1.plot(lens_ch0)

# ax1.set_ylim(-0.1, 1.1)
# plt.close(fig1)
# fig1.tight_layout()
# plt.show()

# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(x + 50, a, '.')
# plt.plot(x + 50, b, '.')
# plt.plot(c, '.')
# # plt.title()
# fig1.tight_layout()
# plt.show()


# hist/bar plot ##############################################################
# hists, bins = np.hist(δt0,100)
# size = 9
# fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
# ax2 = fig2.add_subplot(111)
# fig2.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Country', fontsize=28, labelpad=80,)
# ax2.set_ylabel('Money (M$)', fontsize=28)
# plt.bar(1, 500, color=cs['ggred'])
# plt.bar(2, 1000, color=cs['ggblue'])
# plt.bar(3, 1275, color=cs['mnk_green'])
# plt.bar(4, 10000, color=cs['ggpurple'])
# ax2.set_xlim(0.5, 4.5)
# ax2.set_ylim(0, 11000)
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# size = 4
# fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
# ax1 = fig1.add_subplot(111)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax2.set_xlabel('Δt (ps)')
# ax2.set_ylabel('freq #')
# plt.hist(δt0, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.8)
# plt.hist(δt1, bins=100, edgecolor=cs['mnk_dgrey'], alpha=0.5)

# xyz plot ###################################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# scattter = ax3.scatter(*coords, z, '.', alpha=0.4,
#                       color=cs['gglred'], label='')
# contour = ax3.contour(*coords, z, 10, cmap=cm.jet)
# surface = ax3.plot_surface(*coords, z, 10, cmap=cm.jet)
# wirefrace = ax3.plot_wireframe(*coords, z, 10, cmap=cm.jet)
# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)
# plt.tight_layout()
# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

# img plot ###################################################################
# ax4, fig4, cs = set.figure('image', 'x axis', 'y axis')
# im4 = plt.imshow(Z, cmap='magma', extent=extents(y) +
#                  extents(x),vmin=0,vmax=100)
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb4 = fig4.colorbar(im4, cax=cax)

# save plot ###################################################################
# ax1.figure.savefig('spec.svg')
# plot_file_name = plot_path + 'plot2.png'
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
# PPT_save_2d(fig1, ax1, 'spec')
