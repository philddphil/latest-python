##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Some defs
##############################################################################
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


# set rcParams for nice plots ################################################
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


# Find the elements immediately after the clock reset occurs #################
def find_clk_resets(a):
    b = []
    for i0, v0 in enumerate(a):
        if v0 < 0:
            b.append(i0)
    return b


# Unwrap data to account for clock resets ####################################
def unwrap_data(data, resets):
    DATA = []
    ni = 0
    for i0, v0 in enumerate(resets):
        DATA.append(data[ni:v0])
        ni = resets[i0] + 1
    return DATA


# Find the clock tick before the channel arrival time (ch_t) #################
def find_tick(ch_t, clk_tmp):
    t_bin = int(np.floor(ch_t / 10000))

    # handles the case where the click in the channel occur after the last
    # clock tick of the cycle
    if t_bin + 1 >= len(clk_tmp):
        t_bin = len(clk_tmp) - 1

    clk_t = clk_tmp[t_bin]
    # handles a rounding error which results in the use of the next clock tick
    # rather than the last clock tick
    dt = (ch_t - clk_t) / 10
    if dt < 0:
        dt = dt + 1000

    return dt, t_bin


# This function histograms the arrival times and finds the maximum ###########
def check_delay(DATAS, i0, i1):
    dts = np.zeros((len(DATAS[i0][i1]), 1))
    clk_tmp = DATAS[4][i1]
    for j0 in np.arange(len(DATAS[i0][i1])):
        ch_t = DATAS[i0][i1][j0]
        dt, t_bin = find_tick(ch_t, clk_tmp)
        dts[j0] = dt
    histmin = 0
    histmax = 1000
    histn = 1000
    edges = np.linspace(histmin, histmax, histn)
    N, edges = np.histogram(dts, edges)
    Idx = np.argmax(N)
    delay = np.round(edges[Idx])
    window = [delay - 15, delay + 15]
    return window


# Build of an array of coincidence occurances ################################
def count_coincidence(DATAS, i0, i1, coinc, window):
    dts_ct = []
    ch_ts_filt = []
    dts = np.zeros((len(DATAS[i0][i1]), 1))
    t_bin_last = 0
    clk_tmp = DATAS[4][i1]
    for j0 in np.arange(len(DATAS[i0][i1])):
        ch_t = DATAS[i0][i1][j0]
        dt, t_bin = find_tick(ch_t, clk_tmp)
        dts[j0] = dt

        if dt > window[0] and dt < window[1] and t_bin != t_bin_last:
            coinc[t_bin] = coinc[t_bin] + 1
            dts_ct = [dts_ct, dt]
            ch_ts_filt.append(ch_t)

    t_bin_last = t_bin
    return coinc, ch_ts_filt


##############################################################################
# Do some stuff
##############################################################################
# Specify directory and datasets
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211")
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212"
      r"\g4_1MHzPQ_48dB_cont_snippet_3e6")
# d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212\
#   g4_1MHzTxPIC_55dB_cont_snippet_3e6")
d1 = d0 + r'\Py data'

f1 = d0 + r'\1.txt'
f2 = d0 + r'\2.txt'
f3 = d0 + r'\3.txt'
f4 = d0 + r'\4.txt'
f6 = d0 + r'\6.txt'
os.chdir(d0)

fs = [f1, f2, f3, f4, f6]
# Laser rep rate in Hz
Laser_rep_rate = 1e6
chns = 4
print('# of channels', chns)
# Deal with the clock channel seperately to get the offset
data6 = np.genfromtxt(f6)

# Time of the experiment in s
t_tot = np.shape(data6)[0] / Laser_rep_rate
clk_ticks = np.shape(data6)[0]

# Number of time bins for t series
timebins_N = np.shape(data6)[0] * 10000

offset = np.min(data6)
clk = data6 - offset

# Loop over data sets, unwrap around clock resets
# initialise list of data to unwrap
DATAS = []

# loop through data set
for i0, v0 in enumerate(fs):
    # import channel data
    data = np.genfromtxt(v0) - offset
    # differentiate to get reset points
    ddata = np.diff(data)
    # locate clock reset values
    resets = find_clk_resets(ddata)
    cycles = len(resets)
    print('# of clock cycles', cycles)
    # unwrap data from n * various length vectors to a list of lists
    DATAS.append(unwrap_data(data, resets))

coincs_all = []
Ts_filt = DATAS
# loop over clock cycles
for i0, v0 in enumerate(np.arange(cycles)):
    print(i0)
    t_cycle = len(DATAS[4][i0])
    coinc = np.zeros((t_cycle, 1))
    window = check_delay(DATAS, 1, i0)
    # loop over channels
    for i1, v1 in enumerate(np.arange(chns)):
        print(i1)
        coinc, ch_ts_filt = count_coincidence(DATAS, i1, i0, coinc, window)
        Ts_filt[i1][i0] = ch_ts_filt

    coincs_all = np.append(coincs_all, coinc)

# Save datasets
os.chdir(d1)
for i0, v0 in enumerate(DATAS):
    for i1, v1 in enumerate(DATAS[i0]):
        fname = 'ch' + str(i0) + ' data' + str(i1)
        np.savetxt(fname, Ts_filt[i0][i1])
