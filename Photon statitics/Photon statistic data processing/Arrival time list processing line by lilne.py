##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import combinations
from itertools import islice


##############################################################################
# Some defs
##############################################################################
# Save 2d plot with a colourscheme suitable for ppt, as a png ################
def PPT_save_2d(fig, ax, name):

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
            ax.figure.savefig(name)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no))
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


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


# Set up figure for plotting #################################################
def start_stop(starts, stops, dydx1, t_range, i0, d3, glob_dts, dt_chs):
    loc = 100
    # pad stop channel for region searching
    tt_pad = np.pad(stops, loc + 1)
    # loop through tt0_ns as our start channel
    for i1, v1 in enumerate(starts):
        # print(i1)
        # trunkate full tt1_ns list to a region of 2 * locale values around
        # the same time as the value v1 is (need to use dydx1 to convert)

        # 1. find the corresponding index in tt1_ns
        i_tt1 = int(v1 / dydx1)

        # 2.  specify locale around idx to check - get both times & idx vals
        tt_local = tt_pad[i_tt1:i_tt1 + 2 * loc]
        x_local = np.arange(i_tt1, i_tt1 + 2 * loc) - loc + 1

        # substract ith value of tt0_ns (v1) from tt1_ns & use abs operator
        tt_temp = np.abs(tt_local - v1)
        # find idx of min
        try:
            dt_idx = np.argmin(tt_temp)
        except:
            break

        # find time difference of min value
        dt = tt_local[dt_idx] - v1

        # check if value is of interest
        if -t_range < dt < t_range:
            glob_dts.append(dt)

    (q, r) = divmod(i0, 1000)
    if r == 0 and q != 0:
        os.chdir(d3)
        print('saving dts')
        dt_file = dt_chs + str(q - 1)
        np.save(dt_file, np.asarray(glob_dts))
        print('length = ', len(glob_dts))
        glob_dts = []
    return q, glob_dts


# Run start-stop to get dt list & save #######################################
def gen_dts_from_tts(d2, TCSPC, t_lim=100000, chA='ch0', chB='ch1'):
    d1 = os.path.split(d2)[0] + r"\arrival time files"
    start_time = time.time()
    
    os.chdir(d1)

    datafiles0 = glob.glob(d1 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d1 + r'\*' + chB + r'*')
    
    datafiles = [datafiles0,datafiles1]
    datafiles.sort(key=len)

    d3 = d2 + r"\dts " + chA + " & " + chB
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]

    try:
        os.mkdir(d3)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)
    
    
    os.chdir(d3)
    # define a ROI range to check for co-incidences over
    glob_dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0

    for i0, v0 in enumerate(datafiles[0]):
        dt_chs = 'dts for chs ' + a + b + ' ' + str(i0) + ' '
        os.chdir(d1)
        # print output for keeping track of progress
        print('calc ', dt_chs, ' file', i0, 'of', len(datafiles[0]))

        TT0 = np.load(datafiles0[i0], allow_pickle=True)
        TT1 = np.load(datafiles1[i0], allow_pickle=True)

        TTs = [TT0,TT1]
        TTs.sort(key=len)

        for i1, v1 in enumerate(TTs[0]):
            (q, r) = divmod(i1, 100)
            if r == 0:
                print_time = time.time() - start_time
                print('clock span', i1, 'of', len(TTs[0]), round(print_time,2))
            # convert to ns
            # note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt0 = [j0 * 1e2 for j0 in TT0[i1]]
                tt1 = [j0 * 1e2 for j0 in TT1[i1]]
            elif TCSPC == 'FCT':
                tt0 = [j0 * 1e-1 for j0 in TT0[i1]]
                tt1 = [j0 * 1e-1 for j0 in TT1[i1]]
            else:
                print('Choose hardware, HH or FCT')
                break

            # calc total time and # counts, count rates & gradient functions
            tot_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-9
            global_t += tot_t
            c0 = len(tt0)
            c1 = len(tt1)

            cps0 = c0 / tot_t
            cps1 = c1 / tot_t

            global_cps0.append(cps0)
            global_cps1.append(cps1)

            dydx0 = np.max(tt0) / len(tt0)
            dydx1 = np.max(tt1) / len(tt1)

            #######################################################################
            # Perform start-stop measurements
            #######################################################################
            q, glob_dts = start_stop(
                tt0, tt1, dydx1, t_lim, i1, d3, glob_dts, dt_chs)

    # print('saving final dts')
    os.chdir(d3)
    dt_file = dt_chs + str(q - 1)
    np.save(dt_file, np.asarray(glob_dts))
    print('final save')
    global_dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)

    ##########################################################################
    # Save global Histogram values & info
    ##########################################################################
    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')


# Generate histogram vals and bins from dt list & save #######################
def gen_hist_cvs_from_dts(d2, res=0.4, t_range=25100, chA='ch0', chB='ch1'):
    d3 = d2 + r"\dts " + chA + " & " + chB
    os.chdir(d3)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]

    datafiles = glob.glob(d3 + r'\dts_chs' + a + b + r'_*')
    print(d3 + r'\dts_chs' + a + b + r'_*')
    print(len(datafiles))
    res = 0.4
    t_range = 25100
    nbins = int(2 * t_range / res + 1)
    edges = np.linspace(-t_range, t_range, nbins + 1)
    hists = np.zeros(nbins)

    for i0, v0 in enumerate(datafiles[0:]):
        print('saving hist & bins csv - ', i0, 'of', len(datafiles))
        dts = np.genfromtxt(v0)
        hist, bin_edges = np.histogram(dts, edges)
        hists += hist

    np.savetxt("g2_hist.csv", hists, delimiter=",")
    np.savetxt("g2_bins.csv", bin_edges, delimiter=",")


# Plot g2 from histogram of counts ###########################################
def g2_plot_from_hist_cvs(d2, xlim=1000, chA='ch0', chB='ch1'):
    d3 = d2 + r"\dts " + chA + " & " + chB
    f0 = d3 + r"\g2_hist.csv"
    f1 = d3 + r"\g2_bins.csv"
    f2 = d3 + r"\other_global.csv"

    hist = np.genfromtxt(f0)
    bin_edges = np.genfromtxt(f1)

    bin_w = (bin_edges[1] - bin_edges[0]) / 2
    print('max hist value:', np.max(hist))

    ts = np.linspace(bin_edges[1], bin_edges[-1] -
                     bin_w, len(bin_edges) - 1)

    total_t, ctsps_0, ctsps_1 = np.genfromtxt(f2)
    g2s = hist / (ctsps_0 * ctsps_1 * 1e-9 * total_t * 2 * bin_w)

    print('total cps:', np.round(ctsps_0 + ctsps_1))

    ##########################################################################
    # Fits to data
    ##########################################################################
    ts_fit = np.linspace(ts[0], ts[-1], 500000)
    a = (ctsps_0 + ctsps_1) * 1e-9
    decay_exp = np.exp(-1 * np.abs(ts_fit * a / 3))
    ##########################################################################
    # Plot data
    ##########################################################################

    os.chdir(d3)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='g2s', size=4)
    plt.title(chA + ' & ' + chB)
    ax1.set_xlim(-1 * xlim, xlim)

    ax1.plot(ts, g2s,
             '.-', markersize=5,
             lw=0.5,
             alpha=1, label='')
    ax1.plot(ts_fit, decay_exp)
    # ax1.set_yscale('log')
    # plt.show()
    plotname = 'g2s'
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)


# Call proc_n_lines to process a .lst file into four #py.txt arrival times ###
def proc_lst(d0):
    os.chdir(d0)
    f0 = d0 + r"\TEST.lst"

    tot_lines = sum(1 for line in open(f0))
    print(tot_lines)
    # create files to write to
    f1 = open('ch1.txt', 'w+')
    f2 = open('ch2.txt', 'w+') 
    f3 = open('ch3.txt', 'w+')    
    f4 = open('ch4.txt', 'w+') 

    data_start = tot_lines
    scale = 16
    current_line = 1
    with open(f0, 'r') as input_file:
        for line in input_file:
            current_line += 1           
            if '[DATA]' in line:
                data_start = current_line
            if current_line > data_start:
                missed_counts = 0
                line_hex = line.rstrip('\n')
                line_int = int(line_hex, scale)
                line_bin = bin(line_int).zfill(8)
                ch_bin = line_bin[-3:]
                ch_int = int(ch_bin, 2)
                t_bin = line_bin[0:-4]
                try:
                    t_int = int(t_bin, 2)
                    if ch_int == 1:
                        f1.write(str(t_int) + '\n')
                    if ch_int == 2:
                        f2.write(str(t_int) + '\n')
                    if ch_int == 3:
                        f3.write(str(t_int) + '\n')
                    if ch_int == 6:
                        f4.write(str(t_int) + '\n')                    
                except:
                    missed_counts += 1
                    print(missed_counts, '@', line, 'hex ', line_hex)
                    pass
    f1.close()
    f2.close()
    f3.close()
    f4.close()


# Find the elements immediately after the clock reset occurs #################
def find_clk_resets(a):
    b = []
    for i0, v0 in enumerate(a):
        if v0 < -10000:
            b.append(i0)
    return b


# Unwrap data to account for clock resets ####################################
def unwrap_data(data, resets):
    # This function takes a slice of a large dataset, length N and reshapes 
    # clock resets at indices given by 'resets' into a list of P clock resets
    # each of the P clock resets may have a different length, but the sum is N

    ni = 0
    datas = []
    for i0, v0 in enumerate(resets):
        datas.append(data[ni:v0 + 1])
        ni = v0 + 1
    return datas


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


# Unwrap multiple arrival time files into npy arrays in a new dir.  ##########
def unwrap_4ch_data(d0):
    d1 = d0 + r"\Py data"
    d2 = d1 + r"\arrival time files"

    f1 = d0 + r'\ch1.txt'
    f2 = d0 + r'\ch2.txt'
    f3 = d0 + r'\ch3.txt'
    f4 = d0 + r'\ch4.txt'

    os.chdir(d2)

    fs = [f1, f2, f3, f4]

    # Loop over data sets, unwrap around clock resets
    

    for i0, v0 in enumerate(fs):
        print('channel', i0, 'file', v0)
        tot_lines = sum(1 for line in open(v0))
        print('total lines', tot_lines)
        with open(v0, 'r') as input_file:
            last_data = 0
            curr_line = 0
            data_array = []
            data_row = []
            for line in input_file:
                curr_line += 1
                print(curr_line, 'in', tot_lines)
                data = int(line)
                ddata = data - last_data
                
                if ddata >= 0:
                    data_row.append(data)

                else:
                    data_array.append(data_row)
                    data_row = []

                # differentiate to get reset points
                last_data = data 
                
            # save 2d array of arrival times  to file
            fname = 'ch' + str(i0)
            np.save(fname, data_out)


# Prepare the directories and channel names
def prep_dirs_chs(d0):
    d1 = d0 + r'\py data'
    d2 = d1 + r'\time difference files'
    d3 = d1 + r"\arrival time files"

    try:
        os.mkdir(d1)
    except OSError:
        print("Creation of the directory %s failed" % d1)
    else:
        print("Successfully created the directory %s " % d1)

    try:
        os.mkdir(d2)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    try:
        os.mkdir(d3)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    Chs = ['ch0', 'ch1', 'ch2', 'ch3']
    Chs_combs = list(set(combinations(Chs, 2)))
    return d1, d2, d3, Chs_combs


##############################################################################
# Import data (saved by python code filter data.py)
##############################################################################
d0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0"
d1, d2, d3, Chs_combs = prep_dirs_chs(d0)

# proc_lst(d0)
unwrap_4ch_data(d0)

# for i0, v0 in enumerate(Chs_combs[0:1]):
    # chA, chB = v0[0:2]
    # print('channels:', chA, ' & ', chB)
    # gen_dts_from_tts(d2, 'FCT', 100000, chA, chB)
    # gen_hist_cvs_from_dts(d2, 0.8, 100000, chA, chB)
    # g2_plot_from_hist_cvs(d2, 200, chA, chB)
