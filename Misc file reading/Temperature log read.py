##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


##############################################################################
# Some defs
##############################################################################
#### Custom palette for plotting #############################################
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
               'ggred': [217 / 255, 83 / 255, 25 / 255],
               'ggblue': [30 / 255, 144 / 255, 229 / 255],
               'ggpurple': [145 / 255, 125 / 255, 240 / 255],
               'ggyellow': [229 / 255, 220 / 255, 90 / 255],
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


#### set rcParams for nice plots #############################################
def ggplot_sansserif():
    colours = palette()
    # plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
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


#### Set up figure for plotting ##############################################
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=4):
    ggplot_sansserif()
    cs = palette()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1, cs


#### Save 2d plot with a colourscheme suitable for ppt, as a png #############
def PPT_save_2d(fig, ax, name, dpi=600):

    # Set plot colours
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')

    # Loop to check for file - appends filename with _# if name already exists
    f_exist = True
    app_no = 0
    while f_exist is True:
        if os.path.exists(name + '.png') is False:
            ax.figure.savefig(name, dpi=dpi)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no), dpi=dpi)
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


# Load temperature log ########################################################
def load_T_log(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close
    t_sec = []
    t_date = []
    T = []
    H = []
    for i0, val in enumerate(data[0:]):
        t_string = val.split("\t")[0]
        t_datetime = datetime.strptime(t_string, "%d/%m/%Y %H:%M:%S")
        t_sec = np.append(t_sec, t_datetime.timestamp())
        t_date = np.append(t_date, t_datetime)
        T = np.append(T, float(val.split("\t")[1]))
        try:
            H = np.append(H, float(val.split("\t")[2]))
        except:
            H = np.append(H, 0)
    return t_date, T, H


##############################################################################
# Do some stuff
##############################################################################
p0 = (r"C:\Data\SCM\20210813 T Data")
T_files = glob.glob(p0 + r'\*.txt')
print('number of files', len(T_files))

Hs = []
Ts = []
t_dates = []

for i0, v0 in enumerate(T_files[-5:]):
    print(v0)
    t_date, T, H = load_T_log(v0)
    
    t_dates = np.append(t_dates, t_date, axis=0)
    Ts = np.append(Ts, T, axis=0)
    Hs = np.append(Hs, H, axis=0)


##############################################################################
# Plot some figures
##############################################################################
ax1, fig1, cs = set_figure('Temperatures', 'time', 'Temperature / K',)

ax1a = ax1.twinx()
ax1a.set_ylabel('Heater %', color=cs['ggblue'])

ax1.plot(t_dates, Ts, 'o', color=cs['gglred'], alpha=0.2, label='T points')
ax1.plot(t_dates, Ts, '-', lw=0.5, color=cs['ggdred'], label='T line')
ax1a.plot(t_dates, Hs, '-', lw=0.5, alpha=0.5, color=cs['ggblue'], label='Heater')
plt.tight_layout()
ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
plt.show()
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
os.chdir(p0)
PPT_save_2d(fig1, ax1, 'Ts')
