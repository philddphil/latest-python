##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as opt
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


# Save 2d plot with a colourscheme suitable for ppt, as a png ################
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


# Prepare the directories and channel names ##################################
def prep_dirs_chs(d0, ss_or_win = ' ss'):
    d1 = d0 + r'\time difference files' + ss_or_win

    try:
        os.mkdir(d1)
    except OSError:
        print("Creation of the directory %s failed" % d1)
    else:
        print("Successfully created the directory %s " % d1)

    return d1


# Check if string is number ##################################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Saturation curve ###########################################################
def I_sat(x, I_sat, P_sat, P_bkg, bkg):
    y = (I_sat * x) / (P_sat + x) + P_bkg * x + bkg
    return y


# Plot the I sat curve for data stored in file. Note plt.show() ##############
def I_sat_plot(file, directory=None, title=''):
    if directory is None:
        head, tail = os.path.split(v0)
        directory = tail

    data = np.genfromtxt(file, delimiter=',', skip_header=0)
    Ps = data[0]
    kcps = data[1] / 1000

    initial_guess = (1.5e2, 1e-1, 1e3, 1e0)
    popt, _ = opt.curve_fit(I_sat, Ps, kcps, p0=initial_guess,
                            bounds=((0, 0, 0, 0),
                                    (np.inf, np.inf, np.inf, np.inf)))

    Ps_fit = np.linspace(np.min(Ps), np.max(Ps), 1000)
    Isat_fit = I_sat(Ps_fit, *popt)

    Sat_cts = np.round(popt[0])
    P_sat = np.round(popt[1] * 1e3, 3)
    Prop_bkg = np.round(popt[2])
    bkg = np.round(popt[3])

    lb0 = 'fit'
    lb1 = 'I$_{sat}$ = ' + str(Sat_cts) + 'kcps'
    lb2 = 'bkg = ' + str(Prop_bkg) + 'P + ' + str(bkg)
    lb3 = 'I$_{sat}$ = ' + str(Sat_cts) + 'kcps, ' + \
        'P$_{sat}$ = ' + str(P_sat) + 'mW'

    ##########################################################################
    # Plot some figures
    ##########################################################################
    ax1, fig1, cs = set_figure(name='figure',
                               xaxis='Power (mW)',
                               yaxis='kcounts per secound',
                               size=4)

    plt.plot(Ps, kcps, 'o:', label='data')
    plt.plot(Ps_fit, Isat_fit, '-', label=lb0)
    plt.plot(Ps_fit, I_sat(Ps_fit, popt[0], popt[1], 0, popt[3]),
             '--', label=lb1)
    plt.plot(Ps_fit, I_sat(Ps_fit, 0, popt[1], popt[2], popt[3]),
             '--', label=lb2)

    ax1.legend(loc='lower right', fancybox=True, framealpha=1)

    plt.title(lb3)
    plt.tight_layout()
    # plt.show()
    ax1.legend(loc='lower right', fancybox=True,
               facecolor=(1.0, 1.0, 1.0, 0.0))
    os.chdir(directory)
    PPT_save_2d(fig1, ax1, 'Isat')
    plt.close(fig1)
    return Sat_cts, P_sat


# Plot the x/y tracking sweep ################################################
def plot_int_profile(directory):
    csvs = glob.glob(directory + r'\*.csv')
    x_data = np.genfromtxt(csvs[0], delimiter=',')
    x_fit = np.genfromtxt(csvs[1], delimiter=',')
    y_data = np.genfromtxt(csvs[2], delimiter=',')
    y_fit = np.genfromtxt(csvs[3], delimiter=',')

    ax1, fig1, cs = set_figure(name='figure',
                               xaxis='ΔVs',
                               yaxis='cps')
    ax1.plot(x_data[0] - np.mean(x_data[0]), x_data[1], '.',
             c=cs['ggdred'], label='x')
    ax1.plot(x_fit[0] - np.mean(x_fit[0]), x_fit[1], c=cs['gglred'])
    ax1.plot(y_data[0] - np.mean(y_data[0]), y_data[1], '.',
             c=cs['ggdblue'], label='y')
    ax1.plot(y_fit[0] - np.mean(y_fit[0]), y_fit[1], c=cs['gglblue'])

    plt.tight_layout()
    ax1.legend(loc='lower right', fancybox=True,
               facecolor=(1.0, 1.0, 1.0, 0.0))
    PPT_save_2d(fig1, ax1, 'Profiles')
    plt.close(fig1)


# Generate histogram vals and bins from dt list & save #######################
def gen_dts_from_tts(d2, d3, TCSPC, chA='ch0', chB='ch1'):

    os.chdir(d3)
    file_str = r"\dts " + chA + " & " + chB
    d_dt = d2 + file_str

    try:
        os.mkdir(d_dt)
    except:
        pass

    dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0

    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d3 + r'\*' + chB + r'*')
    print(len(datafiles0))
    file_number = min([len(datafiles0), len(datafiles1)])
    dt_file_number = 0
    for i0 in np.arange(file_number):
        print(datafiles0[i0])
        os.chdir(d_dt)
        if TCSPC == 'HH':
            TT0 = np.loadtxt(datafiles0[i0])
            TT1 = np.loadtxt(datafiles1[i0])
            data_loop = enumerate([0])

        if TCSPC == 'FCT':
            TT0 = np.load(datafiles0[i0], allow_pickle=True)
            TT1 = np.load(datafiles1[i0], allow_pickle=True)
            TTs = [TT0, TT1]
            TTs.sort(key=len)
            data_loop = enumerate(TTs[0])

        for i1, v1 in data_loop:

            # convert to ns
            # note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt0 = [j0 * 1e2 for j0 in TT0]
                tt1 = [j0 * 1e2 for j0 in TT1]
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
            dts = closest_val(tt1, tt0, dts)

            if i1 % 10000 == 0:
                dt_file_number += 1
                print('saving dts', dt_file_number)
                dt_file = 'dts ' + chA + ' ' + chB + ' ' + str(dt_file_number)
                np.save(dt_file, np.asarray(dts))
                dts = []

    os.chdir(d_dt)
    dt_file = dt_file = 'dts ' + chA + ' ' + chB + ' ' + 'f'
    np.save(dt_file, np.asarray(dts))
    dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)

    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')


# Function which calculates closest time differences #########################
def closest_val(a, b, dts):
    c = np.searchsorted(a, b)

    for i0, v0 in enumerate(c):
        if v0 < len(a):
            dt0 = b[i0] - a[v0 - 1]
            dt1 = b[i0] - a[v0]
            if np.abs(dt1) >= np.abs(dt0):
                dt = dt0
            else:
                dt = dt1
            dts.append(dt)
        else:
            dt0 = b[i0] - a[v0 - 1]
            dts.append(dt)
    return dts


# Generate histogram vals and bins from dt list & save #######################
def hist_1d(d2, res=0.4, t_range=25100, chA='ch0', chB='ch1'):
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
        dts = np.load(v0, allow_pickle=True)
        hist, bin_edges = np.histogram(dts, edges)
        hists += hist
    hist_file = ("range-" + str(int(t_range)) +
                 "ns res-" + str(int(res * 1e3)) + "ps g2_hist.csv")
    bins_file = ("range-" + str(int(t_range)) +
                 "ns res-" + str(int(res * 1e3)) + "ps g2_bins.csv")
    np.savetxt(hist_file, hists, delimiter=",")
    np.savetxt(bins_file, bin_edges, delimiter=",")
    return bins_file, hist_file

# Generate histogram vals and bins from dt list & save #######################


def hist_1d_fname(d2, res=0.4, t_range=25100, chA='ch0', chB='ch1'):

    hist_file = ("range-" + str(int(t_range)) +
                 "ns res-" + str(int(res * 1e3)) + "ps g2_hist.csv")
    bins_file = ("range-" + str(int(t_range)) +
                 "ns res-" + str(int(res * 1e3)) + "ps g2_bins.csv")
    return bins_file, hist_file


# Plot g2 from histogram of counts ###########################################
def plot_1d_hist(d1, hist, bin_edges, xlim=1000, chA='ch0', chB='ch1'):
    bin_w = (bin_edges[1] - bin_edges[0]) / 2
    print('max hist value:', np.max(hist))

    ts = np.linspace(bin_edges[1], bin_edges[-1] -
                     bin_w, len(bin_edges) - 1)

    ##########################################################################
    # Plot data
    ##########################################################################

    os.chdir(d1)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='cts', size=4)
    # plt.title(chA + ' & ' + chB)
    ax1.set_xlim(-1 * xlim, xlim)
    ax1.set_ylim(-0.1 * np.max(hist), 1.1 * np.max(hist))

    ax1.plot(ts, hist,
             '.-', markersize=3,
             lw=0.1,
             alpha=0.2, label='')
    ax1.set_ylim(0, 1.1 * np.max(hist))
    plt.show()
    plotname = 'hist'
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)


def gen_dts_from_tts_windowed(d2, d3, TCSPC,  win, chA='ch0', chB='ch1'):
    os.chdir(d3)
    file_str = r"\dts " + chA + " & " + chB
    d_dt = d2 + file_str

    try:
        os.mkdir(d_dt)
    except:
        pass

    dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0

    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d3 + r'\*' + chB + r'*')

    dt_file_number = 0
    for i0, v0 in enumerate(datafiles0[0:]):
        print(datafiles0[i0])
        os.chdir(d_dt)
        if TCSPC == 'HH':
            TT0 = np.loadtxt(datafiles0[i0])
            TT1 = np.loadtxt(datafiles1[i0])
            data_loop = enumerate([0])

        if TCSPC == 'FCT':
            TT0 = np.load(datafiles0[i0], allow_pickle=True)
            TT1 = np.load(datafiles1[i0], allow_pickle=True)
            TTs = [TT0, TT1]
            TTs.sort(key=len)
            data_loop = enumerate(TTs[0])

        for i1, v1 in data_loop:

            # convert to ns
            # note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt0 = [j0 * 1e2 for j0 in TT0]
                tt1 = [j0 * 1e2 for j0 in TT1]
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
            dts = closest_vals(tt1, tt0, dts, win)

            if i1 % 10000 == 0:
                dt_file_number += 1
                print('saving dts', dt_file_number)
                dt_file = 'dts ' + chA + ' ' + chB + ' ' + str(dt_file_number)
                np.save(dt_file, np.asarray(dts))
                dts = []

    os.chdir(d_dt)
    dt_file = dt_file = 'dts ' + chA + ' ' + chB + ' ' + 'f'
    np.save(dt_file, np.asarray(dts))
    dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)

    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')


# Function which calculates closest time differences #########################
def closest_vals(a, b, dts, win):
    c = np.searchsorted(a, b)
    # print('b', len(a))
    # print('a', len(b))
    # print('c', len(c))

    for i0, v0 in enumerate(c):
        N = 0
        start = b[i0]
        while (v0 - N) >= 0 and (v0 + N) < len(a):
            dt = start - a[v0 - (N + 1)]
            if dt <= win:
                dts.append(dt)
                N += 1
                # print('bwd N', N, dt)
            else:
                N = 0
                break

        while (v0 - N) >= 0 and (v0 + N) < len(a):
            dt = start - a[v0 + N]
            if dt >= -1 * win:
                dts.append(dt)
                N += 1
                # print('fwd N', N, dt)
            else:
                break
    return dts


def load_hist_bins(d1, g2_bins_file, g2_hist_file, chA='ch0', chB='ch1'):
    d2 = d1 + r"\dts " + chA + " & " + chB
    f0 = d2 + r'\\' + g2_hist_file
    f1 = d2 + r'\\' + g2_bins_file
    hist = np.genfromtxt(f0)
    bin_edges = np.genfromtxt(f1)
    return hist, bin_edges


def g2_from_cts(d1, hist, bin_edges, chA='ch0', chB='ch1'):
    d2 = d1 + r"\dts " + chA + " & " + chB
    f2 = d2 + r"\other_global.csv"
    total_t, ctsps_0, ctsps_1 = np.genfromtxt(f2)
    print('total cps:', np.round(ctsps_0 + ctsps_1))

    bin_w = bin_edges[1] - bin_edges[0]
    g2s = hist / (ctsps_0 * ctsps_1 * 1e-9 * total_t * bin_w)
    return g2s


# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope and multiple emitters ######################
def g2_fit_ss_0(t, dt, a, b, c, d, z):
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    g3 = (g1 + g2)
    g4 = g3 * (1 - 2 * z * (1 - z)) + 2 * z * (1 - z)
    g5 = g4 * np.exp(-d * np.abs(t - dt))
    return g5

# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope and background #############################


def g2_fit_ss_1(t, dt, a, b, c, d, r):
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    g3 = (g1 + g2)
    g4 = g3 * (r**2) + (1 - r**2)
    g5 = g4 * np.exp(-d * np.abs(t - dt))
    return g5

# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope and multiple emitters ######################


def g2_fit_win_0(t, dt, a, b, c, d, z):
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    g3 = (g1 + g2)
    g4 = g3 * (1 - 2 * z * (1 - z)) + 2 * z * (1 - z)
    g5 = g4 * np.exp(-d * np.abs(t - dt))
    return g4

# g2 function taken from "Berthel et al 2015" for 3 level system with #########
# experimental count rate envelope and background #############################


def g2_fit_win_1(t, dt, a, b, c, d, r):
    g1 = 1 - c * np.exp(- a * np.abs(t - dt))
    g2 = (c - 1) * np.exp(- b * np.abs(t - dt))
    g3 = (g1 + g2)
    g4 = g3 * (r**2) + (1 - r**2)
    g5 = g4 * np.exp(-d * np.abs(t - dt))
    return g4


def plot_g2_fits_win(d1, g2s, bin_edges, xlim, mks=1):
    a = 0.5
    b = 0.002
    c = 1.5
    d = 0.00001
    dt = 0
    z = 0.1
    r = 0.5

    rnd = 5

    # generate time axis
    bin_w = bin_edges[1] - bin_edges[0]
    ts = np.linspace(bin_edges[1] +
                     bin_w, bin_edges[-1] -
                     bin_w, len(bin_edges) - 1)
    ts_fit = np.linspace(ts[0], ts[-1], 1000000)

    init_g = [dt, a, b, c, d, z]

    popt_g0, pcov_g0 = curve_fit(g2_fit_win_0,
                                 ts, g2s, p0=[*init_g],
                                 bounds=([-np.inf, 0, 0, 0, 0, 0],
                                         [np.inf, np.inf, np.inf,
                                          np.inf, np.inf, np.inf]),
                                 maxfev=5000)

    print('Fitted params for g2(t) multiple emitters')
    print('dt =', np.round(popt_g0[0], rnd),
          'a =', np.round(popt_g0[1], rnd),
          'b =', np.round(popt_g0[2], rnd),
          'c =', np.round(popt_g0[3], rnd),
          'd =', np.round(popt_g0[4], rnd),
          'z =', np.round(popt_g0[5], rnd))
    print('g2(0) = ',  np.min(g2_fit_win_0(ts_fit, *popt_g0)))

    init_g = [dt, a, b, c, d, r]

    popt_g1, pcov_g1 = curve_fit(g2_fit_win_1,
                                 ts, g2s, p0=[*init_g],
                                 bounds=([-np.inf, 0, 0, 0, 0, 0],
                                         [np.inf, np.inf, np.inf,
                                          np.inf, np.inf, 1]),
                                 maxfev=5000)
    print('Fitted params for g2(t) background')
    print('dt =', np.round(popt_g1[0], rnd),
          'a =', np.round(popt_g1[1], rnd),
          'b =', np.round(popt_g1[2], rnd),
          'c =', np.round(popt_g1[3], rnd),
          'd =', np.round(popt_g1[4], rnd),
          'r =', np.round(popt_g1[5], rnd))
    print('g2(0) = ', np.min(g2_fit_win_1(ts_fit, *popt_g1)))
    
    ##########################################################################
    # Plot data
    ##########################################################################

    os.chdir(d1)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='cts', size=3)
    # plt.title()
    ax1.set_xlim(-1 * xlim, xlim)
    ax1.set_ylim(0, 1.1 * np.max(g2s))

    ax1.plot(ts - popt_g0[0], g2s,
             '.-', 
             label='data',
             markersize=mks,
             lw=0.1,
             alpha=0.8)
    # ax1.plot(ts_fit - popt_g0[0], g2_fit_win_0(
    #     ts_fit, *popt_g0), '-',
    #     color=cs['ggyellow'],
    #     label='fit 0',
    #     alpha=1,
    #     lw=3.0)
    # ax1.plot(ts_fit - popt_g0[0], g2_fit_win_1(
    #     ts_fit, *popt_g1), '-',
    #     color=cs['ggpurple'],
    #     label='fit 1',
    #     alpha=1,
    #     lw=1)
    # plt.show()
    fig1.tight_layout()
    plotname = 'hist'
    ax1.legend(loc='upper left', fancybox=True, framealpha=0.0)
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)


def plot_g2_fits_ss(d1, g2s, bin_edges, xlim, mks=1):
    a = 0.5
    b = 0.002
    c = 1.5
    d = 0.00001
    dt = 0
    z = 0.1
    r = 0.5

    rnd = 5

    # generate time axis
    bin_w = bin_edges[1] - bin_edges[0]
    ts = np.linspace(bin_edges[1] +
                     bin_w, bin_edges[-1] -
                     bin_w, len(bin_edges) - 1)
    ts_fit = np.linspace(ts[0], ts[-1], 1000000)

    init_g = [dt, a, b, c, d, z]

    popt_g0, pcov_g0 = curve_fit(g2_fit_ss_0,
                                 ts, g2s, p0=[*init_g],
                                 bounds=([-np.inf, 0, 0, 0, 0, 0],
                                         [np.inf, np.inf, np.inf,
                                          np.inf, np.inf, 1]))
    
    print('Fitted params for g2(t) second SPS')
    print('dt =', np.round(popt_g0[0], rnd),
          'a =', np.round(popt_g0[1], rnd),
          'b =', np.round(popt_g0[2], rnd),
          'c =', np.round(popt_g0[3], rnd),
          'd =', np.round(popt_g0[4], rnd),
          'z =', np.round(popt_g0[5], rnd))
    print('g2(0) = ', np.min(g2_fit_ss_1(ts_fit, *popt_g0)))    

    init_g = [dt, a, b, c, d, r]

    popt_g1, pcov_g1 = curve_fit(g2_fit_ss_1,
                                 ts, g2s, p0=[*init_g],
                                 bounds=([-np.inf, 0, 0, 0, 0, 0],
                                         [np.inf, np.inf, np.inf,
                                          np.inf, np.inf, 1]),
                                  maxfev=5000)

    print('Fitted params for g2(t) background')
    print('dt =', np.round(popt_g1[0], rnd),
          'a =', np.round(popt_g1[1], rnd),
          'b =', np.round(popt_g1[2], rnd),
          'c =', np.round(popt_g1[3], rnd),
          'd =', np.round(popt_g1[4], rnd),
          'r =', np.round(popt_g1[5], rnd))
    print('g2(0) = ', np.min(g2_fit_ss_1(ts_fit, *popt_g1)))

    ##########################################################################
    # Plot data
    ##########################################################################

    os.chdir(d1)

    # xy plot ################################################################
    ax1, fig1, cs = set_figure(
        name='figure', xaxis='τ, ns', yaxis='g2(τ)', size=2.5)
    # plt.title()
    ax1.set_xlim(-1 * xlim, xlim)
    ax1.set_ylim(0, 1.1 * np.max(g2s))

    ax1.plot(ts - popt_g1[0], g2s,
             '.-', markersize=mks,
             lw=0.1,
             alpha=0.8, label='')
    # ax1.plot(ts_fit - popt_g1[0], g2_fit_ss_0(
    #     ts_fit, *popt_g0), '-',
    #     color=cs['ggyellow'],
    #     label='fit 0',
    #     alpha=1,
    #     lw=3.0)
    # ax1.plot(ts_fit - popt_g1[0], g2_fit_ss_1(
    #     ts_fit, *popt_g1), '-',
    #     color=cs['ggpurple'],
    #     label='fit 1',
    #     alpha=1,
    #     lw=1)
    # plt.show()
    fig1.tight_layout()
    plotname = 'hist'
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)

##############################################################################
# Do some stuff
##############################################################################
# Data directories

d0 = (r"C:\Data\SCM\20210826 SCM Data\Sequences\26Aug21-002")
Peak_dirs = glob.glob(d0 + r"\*")

for i0, v0 in enumerate(Peak_dirs):
    Peak_dir = os.path.split(v0)[-1]

    #### Check to see if directory name is numeric
    if Peak_dir.isnumeric()==True:
        HH_dir = glob.glob(v0 + r"\HH*")[0]
        print(HH_dir[0])
        ##### Prep more directories to organise data
        d1s = prep_dirs_chs(v0, ' ss')
        # d1w = prep_dirs_chs(d0, ' win')

        t_res = 1
        t_range = 1e5
        # # Gen dt lists if needed (takes time!)
        # gen_dts_from_tts_windowed(d1w, d0, 'HH', t_range)
        print('Calculating dts')
        gen_dts_from_tts(d1s, HH_dir, 'HH')

        # ####### Load windowed/ss datasets
        # try:
        #     bins_file_w, hist_file_w = hist_1d_fname(d1w, t_res, t_range)
        #     hist_w, bin_edges_w = load_hist_bins(d1w, bins_file_w, hist_file_w)
        # except:
        #     bins_file_w, hist_file_w = hist_1d(d1w, t_res, t_range)
        #     hist_w, bin_edges_w = load_hist_bins(d1w, bins_file_w, hist_file_w)

        try:
            bins_file_s, hist_file_s = hist_1d_fname(d1s, t_res, t_range)
            hist_s, bin_edges_s = load_hist_bins(d1s, bins_file_s, hist_file_s)
        except:
            print('Calculating histogram')
            bins_file_s, hist_file_s = hist_1d(d1s, t_res, t_range)
            hist_s, bin_edges_s = load_hist_bins(d1s, bins_file_s, hist_file_s)

        # #### Convert count hists to g2s
        # g2w = g2_from_cts(d1w, hist_w, bin_edges_w)
        g2s = g2_from_cts(d1s, hist_s, bin_edges_s)

        # #### Plot and fit datasets
        # plot_g2_fits_win(d1w, g2w, bin_edges_w, t_range)
        # plot_g2_fits_ss(d1s, g2s, bin_edges_s, t_range)

        ###############################################################################
        ##### Plot some figures
        ###############################################################################
        #### calculate t axis (not bins)
        bin_w = (bin_edges_s[1] - bin_edges_s[0]) / 2

        ts = np.linspace(bin_edges_s[1], bin_edges_s[-1] -
                         bin_w, len(bin_edges_s) - 1)
        os.chdir(v0)
        #### xy plot ####################################################################
        ax1, fig1, cs = set_figure(
            name='figure', 
            xaxis='τ, ns', 
            yaxis='cts', 
            size=4)


        ax1.plot(ts, g2s,
                 '.-', markersize=3,
                 lw=0.1,
                 alpha=0.2, label='')

        # ax1.set_ylim(0, 1.1 * np.max(hist_w))
        plotname = 'hist'
        PPT_save_2d(fig1, ax1, plotname)
        plt.close(fig1)