##############################################################################
# Import some libraries
##############################################################################
import os
import re
import csv
import glob
import numpy as np
from itertools import combinations


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


# Save 2d image with a colourscheme suitable for ppt, as a png ###############
def PPT_save_2d_im(fig, ax, cb, name):
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
    # cbylabel_obj = plt.getp(cb.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='xkcd:black')

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


# For use with extents in imshow #############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


# Run start-stop1,stop2 to get 2 dt list & save ##############################
def gen_dts_from_tts(d2, t_lim=100000, chA='ch0', chB='ch1', chC='ch3'):
    d1 = os.path.split(d2)[0] + r"\arrival time files"

    os.chdir(d1)

    datafiles0 = glob.glob(d1 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d1 + r'\*' + chB + r'*')
    datafiles2 = glob.glob(d1 + r'\*' + chC + r'*')
    d3 = (d2 + r"\dts [" + chA + " & " + chB +
          "] & [" + chA + " & " + chC + ']')
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    c = re.findall('\d+', chC)[0]

    try:
        os.mkdir(d3)
    except OSError:
        print("Creation of the directory %s failed" % d2)
    else:
        print("Successfully created the directory %s " % d2)

    last_file = np.min([len(datafiles0), len(datafiles1), len(datafiles2)])
    dt_chs = 'dts for chs ' + a + b + ' & ' + a + c
    os.chdir(d3)

    # define a ROI range to check for co-incidences over
    locale = 1000
    # define range of τs to be stored
    t_range = t_lim

    global_dts = []
    global_cps0 = []
    global_cps1 = []
    global_cps2 = []
    global_t = 0

    for i0, v0 in enumerate(datafiles0):
        os.chdir(d1)
        # print output for keeping track of progress
        print('calc ', dt_chs, ' file', i0, 'of', len(datafiles0))
        # 1e-7 is the saved resolution - this is 0.1 microsecond
        tta = np.loadtxt(datafiles0[i0])
        ttb = np.loadtxt(datafiles1[i0])
        ttc = np.loadtxt(datafiles2[i0])
        
        # convert channels to ns
        # note the conversion factor is 1e2 for HH & 1e-1 for FCT
        tt0 = [j0 * 1e-1 for j0 in tta]
        tt1 = [j0 * 1e-1 for j0 in ttb]
        tt2 = [j0 * 1e-1 for j0 in ttc]

        # use channel 0 as the start channel
        # subsequent channels can be used as the stop channels
        stop_tts = [tt1, tt2]

        # calc total time and # counts, count rates & gradient functions
        total_t = np.max([np.max(tt0), np.max(tt1), np.max(tt2)]) * 1e-9
        global_t += total_t
        c0 = len(tt0)
        c1 = len(tt1)
        c2 = len(tt2)

        cps0 = c0 / total_t
        cps1 = c1 / total_t
        cps2 = c2 / total_t

        global_cps0.append(cps0)
        global_cps1.append(cps1)
        global_cps2.append(cps2)

        dydx0 = np.max(tt0) / len(tt0)
        dydx1 = np.max(tt1) / len(tt1)
        dydx2 = np.max(tt2) / len(tt2)

        stop_dydxs = [dydx1, dydx2]

        #######################################################################
        # Perform start-stop measurements
        #######################################################################
        fails = 0
        # loop through tt0_ns as our start channel
        for i1, v1 in enumerate(tt0):
            # v1 is the start time
            dts_i = []
            for i2, v2 in enumerate(stop_tts):
                # v2 is the stop array 
                dydxi = stop_dydxs[i2]
                # pad 'stop' channel with 0s
                tt_pad = np.pad(v2, locale + 1)

                # trunkate full tt1_ns list to a region of 2 * locale values 
                # around the same time as the value v1 is (need to use dydx to
                # convert)

                # 1. find the corresponding index in tt1_ns
                i_tti = int(v1 / dydxi)

                # 2.  specify locale around idx to check - get both times & idx
                # vals
                tt_local = tt_pad[i_tti:i_tti + 2 * locale]
                x_local = np.arange(i_tti, i_tti + 2 * locale) - locale + 1

                # substract start time (v1) from stop time & use abs operator
                tt_temp = np.abs(tt_local - v1)

                # find idx of min
                try:
                    dt_idx = np.argmin(tt_temp)
                except:
                    fails += fails
                    break

                # find time difference
                dt = tt_local[dt_idx] - v1
                # if time difference within range, append value
                if - t_range < dt < t_range:
                    dts_i.append(dt)
                    

            if len(dts_i) == 2:
                global_dts.append(dts_i)

        (q, r) = divmod(i0, 10)

        if r == 0 and q != 0:
            os.chdir(d3)
            # print('saving dts')
            dt_file = dt_chs + '_' + str(q - 1) + '.csv'
            with open(dt_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(global_dts)
            # print(dt_file)
            print('dts ', len(global_dts))
            # print(fails)
            global_dts = []

    dt_file = dt_chs + '_' + str(q - 1) + 'final.csv'
    with open(dt_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(global_dts)

    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)
    global_cps2 = np.mean(global_cps2)

    ##########################################################################
    # Save global Histogram values & info
    ##########################################################################
    np.savetxt("other_globals_g3.csv",
               [global_t, global_cps0, global_cps1, global_cps2],
               delimiter=',')


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
    Chs_combs = list(set(combinations(Chs, 3)))
    return d1, d2, d3, Chs_combs


# Histogram 2ddts 
def hists_2d_csvs(d2, res=10, t_range=25010, chA='ch0', chB='ch1', chC='ch2'):
    d4 = (d2 + r"\dts " + chA + " & " + chB +
          " and " + chA + " & " + chC)
    os.chdir(d4)
    datafiles = glob.glob(d4 + r'\dts*')

    nbins = int(2 * t_range / res) + 1
    x_edges = np.linspace(-t_range, t_range, nbins + 1)
    y_edges = np.linspace(-t_range, t_range, nbins + 1)
    dts = []

    data = list(csv.reader(open(datafiles[0])))
    dt1s = []
    dt2s = []
    hists = np.zeros((nbins, nbins))

    hist = []

    for i0, v0 in enumerate(datafiles[0:]):
        data = list(csv.reader(open(datafiles[i0])))
        dt1s = []
        dt2s = []
        print('saving hist & bins csv -', i0, 'of', len(datafiles))
        for i1, v1 in enumerate(data):
            
            if len(v1) == 2:
                dt1s.append(int(float(v1[0])))
                dt2s.append(int(float(v1[1])))
        dt1s_a = np.asarray(dt1s)
        dt2s_a = np.asarray(dt2s)
        # dt1s_int = dt1s_a.astype(np.int32)
        # dt2s_int = dt2s_a.astype(np.int32)
        hist, _, _ = np.histogram2d(dt1s_a, dt2s_a, [x_edges, y_edges])
        hists = hists + hist

    np.savetxt("g3_hist.csv", hists, delimiter=",")
    np.savetxt("g3_xbins.csv", x_edges, delimiter=",")
    np.savetxt("g3_ybins.csv", y_edges, delimiter=",")


##############################################################################
# Import data (saved by python code filter data.py)
##############################################################################
d0 = r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717\1"

d1, d2, d3, Chs_combs = prep_dirs_chs(d0)
print(Chs_combs)

for i0, v0 in enumerate(Chs_combs):
    chA, chB, chC = v0
    print(v0)
    # gen_dts_from_tts(d2, 100000, chA, chB, chC)
    hists_2d_csvs(d2, res=10, t_range=25010, chA='ch0', chB='ch1', chC='ch2')