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
from itertools import combinations

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


# Call proc_n_lines to process a .lst file into four #py.txt arrival times ###
def proc_lst(d0):
    os.chdir(d0)
    f0 = d0 + r"\TEST.lst"

    # create files to write to
    f1 = open('ch1.txt', 'w+')
    f2 = open('ch2.txt', 'w+')
    f3 = open('ch3.txt', 'w+')
    f4 = open('ch4.txt', 'w+')

    data_start = 1000
    scale = 16
    current_line = 1
    with open(f0, 'r') as input_file:
        for line in input_file:
            current_line += 1
            if '[DATA]' in line:
                data_start = current_line
                print('Data starts at line', current_line)
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
        with open(v0, 'r') as input_file:
            tot_times = 0
            previous_data = 0
            curr_line = 0
            array_size = 0
            array_number = 0
            data_array = []
            data_row = []
            for line in input_file:
                curr_line += 1
                data = int(line)
                ddata = data - previous_data

                if ddata >= 0:
                    data_row.append(data)

                elif ddata < 0:
                    data_array.append(data_row)
                    tot_times += len(data_row)
                    array_size += 1
                    data_row = [data]
                previous_data = data

                # save array of arrival times to file
                if array_size == 1000000:
                    fname = 'ch' + str(i0) + ' ' + str(array_number)
                    np.save(fname, data_array)
                    data_array = []
                    array_number += 1

            # save final array of arrival times to file
            fname = 'ch' + str(i0) + ' f'
            np.save(fname, data_array)
            print('total times', tot_times)
            print('total resets', len(data_array))


# Generate histogram vals and bins from dt list & save #######################
def gen_dts_from_tts(d2, d3, TCSPC, t_lim=100000, chA='ch0', chB='ch1'):

    os.chdir(d3)
    file_str = r"\dts " + chA + " & " + chB
    d_dt = d2 + file_str

    try:
        os.mkdir(d_dt)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0

    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d3 + r'\*' + chB + r'*')

    file_number = min([len(datafiles0), len(datafiles1)])

    for i0 in np.arange(file_number):

        os.chdir(d_dt)
        TT0 = np.load(datafiles0[i0], allow_pickle=True)
        TT1 = np.load(datafiles1[i0], allow_pickle=True)

        TTs = [TT0, TT1]
        TTs.sort(key=len)

        for i1, v1 in enumerate(TTs[0]):
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

            tot_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-9
            global_t += tot_t
            c0 = len(tt0)
            c1 = len(tt1)

            cps0 = c0 / tot_t
            cps1 = c1 / tot_t

            global_cps0.append(cps0)
            global_cps1.append(cps1)

            # calculate closest values
            dts = closest_val(tt0, tt1, dts, t_lim)

            (q, r) = divmod(i1, 100000)
            if r == 0:
                print('saving dts')
                dt_file = 'dts ' + chA + ' ' + chB + ' ' + str(q)
                np.save(dt_file, np.asarray(dts))
                dts = []

    # print('saving final dts')
    os.chdir(d_dt)
    dt_file = dt_file = 'dts ' + chA + ' ' + chB + ' ' + 'f'
    np.save(dt_file, np.asarray(dts))
    print('final save')
    dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)

    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')


# Function which calculates closest time differences
def closest_val(a, b, dts, t_lim):
    c = np.searchsorted(a, b)
    for i0 in c:
        if i0 > 0 and i0 < len(a) and i0 < len(b):
            dt0 = b[i0] - a[i0 - 1]
            dt1 = b[i0] - a[i0]
            dt = np.min([dt0, dt1])
            if -t_lim < dt < t_lim:
                dts.append(dt)
        else:
            pass
    return dts


# Generate histogram vals and bins from dt list & save #######################
def gen_hist_cvs_from_dts(d2, res=0.4, t_range=25100, chA='ch0', chB='ch1'):
    d3 = d2 + r"\dts " + chA + " & " + chB
    os.chdir(d3)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    file_str = (r'\dts ' + chA + ' ' + chB)
    print(file_str)
    datafiles = glob.glob(d3 + file_str + r'*')
    print(len(datafiles))

    nbins = int(2 * t_range / res + 1)
    edges = np.linspace(-t_range, t_range, nbins + 1)
    hists = np.zeros(nbins)

    for i0, v0 in enumerate(datafiles[0:]):
        print('saving hist & bins csv - ', i0, 'of', len(datafiles))
        dts = np.load(v0)
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
    # Plot data
    ##########################################################################

    os.chdir(d3)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='g2s', size=4)
    plt.title(chA + ' & ' + chB)
    ax1.set_xlim(-1 * xlim, xlim)

    ax1.plot(ts, hist,
             '.-', markersize=5,
             lw=0.5,
             alpha=1, label='')
    # plt.show()
    plotname = 'g2s'
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)



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


##########################################################################
# Import data (saved by python code filter data.py)
##########################################################################
d0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\0"
d1, d2, d3, Chs_combs = prep_dirs_chs(d0)
unwrap_4ch_data(d0)
# for i0, v0 in enumerate(Chs_combs):
    # chA, chB = v0[0:2]
    # print('channels:', chA, ' & ', chB)
    # gen_dts_from_tts(d2, d3,'FCT', 100000, chA, chB)
    # gen_hist_cvs_from_dts(d2, 0.8, 100000, chA, chB)
    # g2_plot_from_hist_cvs(d2, 200, chA, chB)
