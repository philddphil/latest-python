##############################################################################
# Import some libraries
##############################################################################
import os
import re
import glob
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################################################
# Some defs
##############################################################################
#### Prepare the directories and channel names ###############################
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


#### Call proc_n_lines to process a .lst file into four ascii files ##########
def proc_lst(d0):
    os.chdir(d0)
    f0 = d0 + r"\TEST.lst"

    # create files to write to
    f1 = open('ch1.txt', 'w+')
    f2 = open('ch2.txt', 'w+')
    f3 = open('ch3.txt', 'w+')
    f4 = open('ch4.txt', 'w+')

    # initialise parameters
    # code will check up to the first 1000 lines for [DATA] str
    data_start = 1000
    scale = 16
    current_line = 0

    # with open syntax to close file automatically
    with open(f0, 'r') as input_file:

        # for loop to go through file line by line
        for line in input_file:
            current_line += 1

            # Find start of data
            if '[DATA]' in line:
                data_start = current_line
            if current_line > data_start:
                missed_counts = 0
                line_hex = line.rstrip('\n')

                # Try/except to parse Hex
                try:
                    line_int = int(line_hex, scale)
                    line_bin = bin(line_int).zfill(8)
                    ch_bin = line_bin[-3:]
                    ch_int = int(ch_bin, 2)
                    t_bin = line_bin[0:-4]
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
                    pass

            # close and reopen files every 1e7
            # this provides text to show code is working
            if current_line % 1e7 == 0:
                print('Processed', current_line, 'lines')
                print('Missed', missed_counts, 'counts')
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f1 = open('ch1.txt', 'a')
                f2 = open('ch2.txt', 'a')
                f3 = open('ch3.txt', 'a')
                f4 = open('ch4.txt', 'a')

    f1.close()
    f2.close()
    f3.close()
    f4.close()


#### Unwrap multiple arrival time files into npy arrays in a new dir.  #######
def unwrap_4ch_data(d0):
    # create directory paths for generated data
    d1 = d0 + r"\Py data"
    d2 = d1 + r"\arrival time files"

    # create directory paths for parsed hex arrival times
    f1 = d0 + r'\ch1.txt'
    f2 = d0 + r'\ch2.txt'
    f3 = d0 + r'\ch3.txt'
    f4 = d0 + r'\ch4.txt'

    os.chdir(d2)

    fs = [f1, f2, f3, f4]

    # Loop over each channel, unwrap around clock resets
    for i0, v0 in enumerate(fs):

        print('channel', i0, 'file', v0)

        # open input file for reading (with auto-close)
        with open(v0, 'r') as input_file:

            # initialise values
            non_increases = 0
            tot_times = 0
            total_resets = 0
            previous_data = 0
            curr_line = 0
            array_size = 0
            array_number = 0
            data_array = []
            data_row = []

            # loop over lines in input file
            for line in input_file:
                curr_line += 1
                data = int(line)
                ddata = data - previous_data

                # append time to data_row if greater than previous time
                if ddata >= 0:
                    data_row.append(data)

                # catch clock resets, when the difference is < -1e6
                elif ddata < -1e6:
                    data_array.append(data_row)
                    tot_times += len(data_row)
                    array_size += 1
                    data_row = [data]

                # catch times that are not added to data as they are
                # non-increasing values
                else:
                    non_increases += 1

                previous_data = data

                # save array of arrival times to file
                # Set this to 10000 for laptop processing
                # Or 1000000 for server processing
                if array_size == 10000:
                    fname = 'ch' + str(i0) + ' ' + str(array_number)
                    np.save(fname, data_array)

                    array_number += 1
                    print('saving', fname,
                          'phtons arrivals so far', curr_line,
                          '               ',
                          'resets so far', array_number * len(data_array))

                    data_array = []
                    array_size = 0

            # save final array of arrival times to file
            fname = 'ch' + str(i0) + ' f'
            np.save(fname, data_array)
            print('total times', curr_line)
            print('total resets', len(data_array))
            write_str = ('ch arrivals ' +
                         str(i0) + '\n'
                         'Total photons = ' +
                         str(curr_line) + '\n'
                         'Total resets = ' +
                         str(array_nuber * len(data_array)) + '\n'
                         'Not counted = ' +
                         str(non_increases) + '\n')
            log.write(write_str)


#### Generate histogram vals and bins from dt list & save ####################
def gen_dts_from_tts(d2, d3, TCSPC, chA='ch0', chB='ch1'):

    #### Generate data directory for dts
    os.chdir(d3)
    file_str = r"\dts " + chA + " & " + chB
    d_dt = d2 + file_str

    try:
        os.mkdir(d_dt)
    except:
        pass

    #### Initialise variables
    dts_number = 0
    dts = []
    global_cps0 = []
    global_cps1 = []
    global_t = 0
    dt_file_number = 0

    #### Generate lists of data files
    datafiles0 = glob.glob(d3 + r'\*' + chA + r'*')
    datafiles1 = glob.glob(d3 + r'\*' + chB + r'*')

    #### Print lengths of files (should always be equal)
    print(chA, len(datafiles0), chB, len(datafiles1))

    #### Loop over data files
    for i0 in np.arange(len(datafiles0)):

        #### Change pwd to data directory 
        os.chdir(d_dt)

        #### Try/except on loading the unwrapped data
        try:
            TT0 = np.load(datafiles0[i0], allow_pickle=True)
            TT1 = np.load(datafiles1[i0], allow_pickle=True)
        except:
            write_str = ("dts " + chA + " & " + chB +
                         ' break at file' + str(i0))
            log.write(write_str)
            print('dt proc. stop at file', i0)
            print(chA, len(TT0), chB, len(TT1))
            break

        #### Compile npy arrays into list of 2 arrays
        TTs = [TT0, TT1]
        print('resets per file a', np.shape(TT0))
        print('resets per file b', np.shape(TT1))
        TTs.sort(key=len)

        #### Loop through the shorter list (should be equal anyway)
        for i1, v1 in enumerate(TTs[0]):

            #### convert to the times to ns
            #### note the conversion factor is 1e2 for HH & 1e-1 for FCT
            if TCSPC == 'HH':
                tt0 = [j0 * 1e2 for j0 in TT0[i1]]
                tt1 = [j0 * 1e2 for j0 in TT1[i1]]
            elif TCSPC == 'FCT':
                tt0 = [j0 * 1e-1 for j0 in TT0[i1]]
                tt1 = [j0 * 1e-1 for j0 in TT1[i1]]
            else:
                print('Choose hardware, HH or FCT')
                break

            #### grab the largest arrival time and add to global time
            tot_t = np.max([np.max(tt0), np.max(tt1)]) * 1e-9
            global_t += tot_t

            #### grab the number of arrivals in each list 
            c0 = len(tt0)
            c1 = len(tt1)

            #### calculate the cps for each reset
            cps0 = c0 / tot_t
            cps1 = c1 / tot_t

            #### append the cps to a global list
            global_cps0.append(cps0)
            global_cps1.append(cps1)

            #### calculate closest values
            dts = closest_val(tt1, tt0, dts)

            #### empty the dts list every 1e4 resets and save
            if i1 % 1e4 == 0:
                dt_file_number += 1
                print('saving dts', dt_file_number)
                dt_file = 'dts ' + chA + ' ' + chB + ' ' + str(dt_file_number)
                np.save(dt_file, np.asarray(dts))
                dts_number += len(dts)
                print('dts so far', dts_number)
                dts = []

    #### save final file and write to log
    os.chdir(d_dt)
    dt_file = dt_file = 'dts ' + chA + ' ' + chB + ' ' + 'f'
    np.save(dt_file, np.asarray(dts))
    dts = []
    global_cps0 = np.mean(global_cps0)
    global_cps1 = np.mean(global_cps1)
    np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
               delimiter=',')
    write_str = ("dts " + chA + " & " + chB + '\n' +
                 'Total dts = ' + str(dts_number) + '\n')
    log.write(write_str)


#### Function which calculates closest time differences ######################
def closest_val(a, b, dts):
    #### use search sorted to find closest element in b to one in a
    #### dt is then b - a, or ch0 - ch1
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
            dt = b[i0] - a[v0 - 1]
            dts.append(dt)
    return dts


#### Generate histogram vals and bins from dt list & save ####################
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
        print('saving 1D hist & bins csv - ', i0, 'of', len(datafiles))
        dts = np.load(v0, allow_pickle=True)
        hist, bin_edges = np.histogram(dts, edges)
        hists += hist

    np.savetxt("g2_hist.csv", hists, delimiter=",")
    np.savetxt("g2_bins.csv", bin_edges, delimiter=",")


#### Generate histogram vals and bins from dt list & save ####################
def hist_2d(d2, res=50, t_range=25100, chA='ch0', chB='ch1', chC='ch2'):
    d3a = d2 + r"\dts " + chA + " & " + chB
    d3b = d2 + r"\dts " + chA + " & " + chC
    os.chdir(d3)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    c = re.findall('\d+', chC)[0]

    file_str_a = (r'\dts ' + chA + ' ' + chB)
    file_str_b = (r'\dts ' + chA + ' ' + chC)
    print(d3a + file_str_a)
    datafiles_a = glob.glob(d3a + file_str_a + r'*')
    datafiles_b = glob.glob(d3b + file_str_b + r'*')

    nbins = int(2 * t_range / res + 1)
    edges = np.linspace(-t_range, t_range, nbins + 1)
    edges_2d = np.transpose(np.meshgrid(edges, edges))
    print(np.shape(edges_2d), 'data file #', len(datafiles_a))
    hists = np.zeros([nbins, nbins])

    for i0, v0 in enumerate(datafiles_a):
        print('saving 2D" hist & bins csv - ', i0, 'of', len(datafiles_a))
        dtsAB = np.load(datafiles_a[i0], allow_pickle=True)
        dtsAC = np.load(datafiles_b[i0], allow_pickle=True)
        data_array = np.transpose(np.array([dtsAB, dtsAC]))
        hist, bin_edges = np.histogramdd(data_array, [edges, edges])
        hists += hist
        print('max counts', np.max(np.max(hists)))

    d4 = d2 + r"\dts " + chA + chB + " & " + chA + chC

    try:
        os.mkdir(d4)
    except:
        pass
    os.chdir(d4)
    hist_csv_name = ("g3_hist_res_" + str(1000 * res) + 'ps' +
                     "_range_" + str(t_range) + ".csv")
    xbins_csv_name = ("g3_xbins_res_" + str(1000 * res) + 'ps' +
                      "_range_" + str(t_range) + ".csv")
    ybins_csv_name = ("g3_ybins_res_" + str(1000 * res) + 'ps' +
                      "_range_" + str(t_range) + ".csv")
    np.savetxt(hist_csv_name, hists, delimiter=",")
    np.savetxt(xbins_csv_name, bin_edges[0], delimiter=",")
    np.savetxt(ybins_csv_name, bin_edges[1], delimiter=",")


# Generate histogram vals and bins from dt list & save #######################
# Select the two shortest times here #########################################
def hist_2d_alt(d2, res=50, t_range=25100, chA='ch0', chB='ch1', chC='ch2'):
    ### generate paths to the directories with the right dts
    d3a = d2 + r"\dts " + chA + " & " + chB
    d3b = d2 + r"\dts " + chA + " & " + chC
    ### change working directory to d2
    os.chdir(d2)

    ### generate names of the dts files being considered 
    file_str_a = (r'\dts ' + chA + ' ' + chB)
    file_str_b = (r'\dts ' + chA + ' ' + chC)

    ### generate lists of all the file paths for the dt files
    datafiles_a = glob.glob(d3a + file_str_a + r'*')
    datafiles_b = glob.glob(d3b + file_str_b + r'*')

    ### generate the log file to check for dt shifts
    log_file = d2 + r"\log.txt"

    ### generate strings for the new dt file names
    chAB = (chA + chB)
    chAC = (chA + chC)

    ### load the appropriate time shifts from the log file
    with open(log_file) as fp:
        for i0, line in enumerate(fp):
            if chAB in line:
                shift_unit = line.split("\t")[1]
                shift_AB = float(shift_unit.split("  ")[0])
            elif chAC in line:
                shift_unit = line.split("\t")[1]
                shift_AC = float(shift_unit.split("  ")[0])
    print('shifts found', shift_AB, shift_AC)

    ### generate the bin parameters for the histogram
    nbins = int(2 * (t_range / res))
    centres = np.linspace(-t_range, t_range, nbins + 1)
    edges = np.linspace(-(t_range + res / 2), (t_range + res / 2), nbins + 2)

    ### generate an empty array of the right size for histogram values
    hists = np.zeros([len(centres), len(centres)])

    ### loop through the start channel dts(datafiles_a)
    for i0, v0 in enumerate(datafiles_a):
        print('saving 2D hist & bins csv - ', i0, 'of', len(datafiles_a))
        
        ### load dt between channel A and B and apply time shift
        dtsAB = np.load(datafiles_a[i0], allow_pickle=True)
        dtsAB_shift = dtsAB - shift_AB
        
        ### load dt between channel A and C and apply time shift
        dtsAC = np.load(datafiles_b[i0], allow_pickle=True)
        dtsAC_shift = dtsAC - shift_AC

        ### calculate the dt between channel B and C 
        dtsCB_shift = dtsAB_shift - dtsAC_shift

        ### select the smallest 2 dts, corresponding to the time between photons 1 & 2
        ### and 2 & 3
        dtx = []
        dty = []
        for i1, v1 in enumerate(dtsAB_shift):
            abc = [dtsAB_shift[i1],dtsAC_shift[i1],dtsCB_shift[i1]]
            ABC = np.abs(abc)
            T = np.argmax(ABC)
            xy = np.delete(abc, T)
            dtx.append(xy[0])
            dty.append(xy[1])

        dtx = np.squeeze(np.asarray(dtx))
        dty = np.squeeze(np.asarray(dty))
        data_array = np.transpose(np.array([dtx, dty]))

        ### histogram the paired dts
        hist, bin_edges = np.histogramdd(data_array, [edges, edges])
        hists += hist
        print('max counts', np.max(np.max(hists)))
        print('bin_edges_out', np.shape(bin_edges))

    ### Construct new directory name
    d4 = d2 + r"\dts " + chA + chB + " & " + chA + chC

    ### Attempt to create new directory
    try:
        os.mkdir(d4)
    except OSError:
        print("Creation of the directory %s failed (may already exist)" % d4)
    else:
        print("Successfully created the directory %s " % d4)

    ### Change pwd to the new dir
    os.chdir(d4)
    print(d4)

    ### Save data
    hist_f = ("range" + str(int(t_range)) +
              "ns res" + str(int(res * 1e3)) + "ps g3_hist_alt.csv")
    xbins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_xbins_alt.csv")
    ybins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_ybins_alt.csv")
    np.savetxt(hist_f, hists, delimiter=",")
    np.savetxt(xbins_f, bin_edges[0], delimiter=",")
    np.savetxt(ybins_f, bin_edges[1], delimiter=",")
    print('saved', hist_f, 'and bins')
    return xbins_f, ybins_f, hist_f


# Generate histogram vals and bins from dt list & save #######################
# Select the two shortest times (abs) here ###################################
def hist_2d_alt_abs(d2, res=50, t_range=25100, chA='ch0', chB='ch1', chC='ch2'):
    ### generate paths to the directories with the right dts
    d3a = d2 + r"\dts " + chA + " & " + chB
    d3b = d2 + r"\dts " + chA + " & " + chC
    ### change working directory to d2
    os.chdir(d2)

    ### generate names of the dts files being considered 
    file_str_a = (r'\dts ' + chA + ' ' + chB)
    file_str_b = (r'\dts ' + chA + ' ' + chC)

    ### generate lists of all the file paths for the dt files
    datafiles_a = glob.glob(d3a + file_str_a + r'*')
    datafiles_b = glob.glob(d3b + file_str_b + r'*')

    ### generate the log file to check for dt shifts
    log_file = d2 + r"\log.txt"

    ### generate strings for the new dt file names
    chAB = (chA + chB)
    chAC = (chA + chC)

    ### load the appropriate time shifts from the log file
    with open(log_file) as fp:
        for i0, line in enumerate(fp):
            if chAB in line:
                shift_unit = line.split("\t")[1]
                shift_AB = float(shift_unit.split("  ")[0])
            elif chAC in line:
                shift_unit = line.split("\t")[1]
                shift_AC = float(shift_unit.split("  ")[0])
    print('shifts found', shift_AB, shift_AC)

    ### generate the bin parameters for the histogram
    nbins = int(t_range / res)
    centres = np.linspace(res / 2, t_range-res / 2, nbins)
    edges = np.linspace(0, t_range, nbins + 1)

    ### generate an empty array of the right size for histogram values
    hists = np.zeros([len(centres), len(centres)])

    ### loop through the start channel dts(datafiles_a)
    for i0, v0 in enumerate(datafiles_a):
        print('saving 2D hist & bins csv - ', i0, 'of', len(datafiles_a))
        
        ### load dt between channel A and B and apply time shift
        dtsAB = np.load(datafiles_a[i0], allow_pickle=True)
        dtsAB_shift = dtsAB - shift_AB
        
        ### load dt between channel A and C and apply time shift
        dtsAC = np.load(datafiles_b[i0], allow_pickle=True)
        dtsAC_shift = dtsAC - shift_AC

        ### calculate the dt between channel B and C 
        dtsCB_shift = dtsAB_shift - dtsAC_shift

        ### select the smallest 2 dts, corresponding to the time between photons 1 & 2
        ### and 2 & 3 - take the abs values
        dtx = []
        dty = []
        for i1, v1 in enumerate(dtsAB_shift):
            abc = [dtsAB_shift[i1],dtsAC_shift[i1],dtsCB_shift[i1]]
            ABC = np.abs(abc)
            T = np.argmax(ABC)
            xy = np.delete(abc, T)
            dtx.append(np.abs(xy[0]))
            dty.append(np.abs(xy[1]))

        dtx = np.squeeze(np.asarray(dtx))
        dty = np.squeeze(np.asarray(dty))
        data_array = np.transpose(np.array([dtx, dty]))

        ### histogram the paired dts
        hist, bin_edges = np.histogramdd(data_array, [edges, edges])
        hists += hist
        print('max counts', np.max(np.max(hists)))
        print('bin_edges_out', np.shape(bin_edges))

    ### Construct new directory name
    d4 = d2 + r"\dts " + chA + chB + " & " + chA + chC

    ### Attempt to create new directory
    try:
        os.mkdir(d4)
    except OSError:
        print("Creation of the directory %s failed (may already exist)" % d4)
    else:
        print("Successfully created the directory %s " % d4)

    ### Change pwd to the new dir
    os.chdir(d4)
    print(d4)

    ### Save data
    hist_f = ("range" + str(int(t_range)) +
              "ns res" + str(int(res * 1e3)) + "ps g3_hist_alt_abs.csv")
    xbins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_xbins_alt_abs.csv")
    ybins_f = ("range" + str(int(t_range)) +
               "ns res" + str(int(res * 1e3)) + "ps g3_ybins_alt_abs.csv")
    np.savetxt(hist_f, hists, delimiter=",")
    np.savetxt(xbins_f, bin_edges[0], delimiter=",")
    np.savetxt(ybins_f, bin_edges[1], delimiter=",")
    print('saved', hist_f, 'and bins')
    return xbins_f, ybins_f, hist_f


#### Plot g2 from histogram of counts ########################################
def plot_1d_hist(d2, xlim=1000, chA='ch0', chB='ch1'):
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
        name='figure', xaxis='τ, ns', yaxis='cts', size=4)
    plt.title(chA + ' & ' + chB)
    ax1.set_xlim(-1 * xlim, xlim)
    ax1.set_ylim(-0.1 * np.max(hist), 1.1 * np.max(hist))

    ax1.plot(ts, hist,
             '.-', markersize=5,
             lw=0.5,
             alpha=1, label='')
    # plt.show()
    os.chdir(d2)
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    plotname = 'hist ' + a + b
    PPT_save_2d(fig1, ax1, plotname)
    plt.close(fig1)


#### Plot a 2d histogram from data in d2 #####################################
def plot_2d_hist(d2, x_lim, res, t_range, chA='ch0', chB='ch1', chC='ch2'):
    hist_csv_name = ("g3_hist_res_" + str(1000 * res) + 'ps' +
                     "_range_" + str(t_range) + ".csv")
    xbins_csv_name = ("g3_xbins_res_" + str(1000 * res) + 'ps' +
                      "_range_" + str(t_range) + ".csv")
    ybins_csv_name = ("g3_ybins_res_" + str(1000 * res) + 'ps' +
                      "_range_" + str(t_range) + ".csv")

    d4 = d2 + r"\dts " + chA + chB + " & " + chA + chC
    print(d4)
    os.chdir(d4)

    hist = np.loadtxt(hist_csv_name, delimiter=',')
    x_edges = np.genfromtxt(xbins_csv_name)
    y_edges = np.genfromtxt(ybins_csv_name)

    bin_w = (x_edges[1] - x_edges[0]) / 2

    ts = np.linspace(x_edges[0] +
                     bin_w, x_edges[-1] -
                     bin_w, len(x_edges) - 1)

    # total_t, ctsps_0, ctsps_1, ctsps_2 = np.genfromtxt(f3)
    # print('time =', total_t)
    # print('cts 0 =', ctsps_0)
    # print('cts 1 =', ctsps_1)
    # print('cts 2 =', ctsps_2)
    # print('bin_w =', 2 * bin_w)

    # # normalise the Glauber function
    # g3s = hist / (ctsps_2 * ctsps_0 * ctsps_1 * 4 * total_t *
    #               (2 * bin_w * 1e-9) ** 2)
    # g3s = g3s
    # print('total cps = ', np.round(ctsps_0 + ctsps_1 + ctsps_2))

    ##########################################################################
    # Plot data
    ##########################################################################

    # profile plots ##########################################################
    hist_x, hist_y = np.shape(hist)
    ax1, fig1, cs = set_figure('profiles', 'time', 'g^3')
    hist_0 = (np.shape(hist)[0] - 1) / 2
    print(hist_0)
    ax1.plot(ts, hist[:, 20], '.--', lw=0.5, markersize=5)
    ax1.plot(ts, np.diag(np.fliplr(hist)), '.--', lw=0.5, markersize=5)
    ax1.set_xlim(-1 * x_lim, x_lim)

    # img plot ###############################################################
    ax4, fig4, cs = set_figure('image', 'x axis', 'y axis')
    ax4.plot(ts, np.ones(len(ts)) * ts[20], lw=1)
    ax4.plot(ts, -ts, lw=1)
    im4 = plt.imshow(hist, cmap='magma',
                     extent=extents(x_edges) + extents(y_edges),
                     vmin=0, vmax=np.max(hist), origin='lower')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig4.colorbar(im4, cax=cax)
    ax4.set_xlim(-1 * x_lim, x_lim)
    ax4.set_ylim(-1 * x_lim, x_lim)
    # plt.show()
    a = re.findall('\d+', chA)[0]
    b = re.findall('\d+', chB)[0]
    c = re.findall('\d+', chC)[0]
    plotname = 'hist ' + a + b + '_' + a + c
    print('max counts', np.max(hist))
    os.chdir(d2)
    # PPT_save_2d(fig1, ax1, 'profiles')
    PPT_save_2d_im(fig4, ax4, cb, plotname)
    # plt.close(fig1)
    plt.close(fig4)


#### Save plot for powerpoint ################################################
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


#### Set up figure for plotting ##############################################
def set_figure(name='figure', xaxis='x axis', yaxis='y axis', size=4):
    ggplot()
    cs = palette()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1, cs


#### For use with extents in imshow ##########################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


#### Save 2d image with a colourscheme suitable for ppt, as a png ############
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


##############################################################################
#### Code to do stuff
##############################################################################
#### specify data directory
d0 = (r"D:\pd10\TEST12 8hr")
os.chdir(d0)

#### set resolutions for final hists
t_res = 0.1
t_range = 1500
t_lim = 200
g3_res = 30 * t_res

#### create log file to write to and open
with open('log.txt', 'w+') as log:
    #### log start
    start_time = time.time()
    #### prepare additional directories for processed data
    d1, d2, d3 = prep_dirs(d0)

    #### call proc_lst
    print('processing lst file into channel arrival time asciis')
    proc_lst(d0)

    #### log lst end
    lst_end = time.time()
    lst_time = int(lst_end - start_time)
    lst_time_str = str(datetime.timedelta(seconds=lst_time))
    write_str = ('lst processing end = ' + lst_time_str + '\n')
    log.write(write_str)
    print("lst proc", lst_time_str)

    #### unwrap ascii arrival times into npy ragged arrays & log time
    print('unwrapping the 4 ascii files into npy arrays')
    unwrap_4ch_data(d0)

    #### log unwrap end
    uw_end = time.time()
    uw_time = int(uw_end - lst_end)
    uw_time_str = str(datetime.timedelta(seconds=uw_time))
    write_str = ('unwrapping end = ' + uw_time_str + '\n')
    log.write(write_str)
    print("uw chs", uw_time_str)

    #### define channel labels
    Chs = ['ch0', 'ch1', 'ch2', 'ch3']

    #### generate list of possible pairings from channels defined above
    Chs_perms2 = list(set(permutations(Chs, 2)))
    #### combinations might be preferable at some point**

    #### this is for logging the timing in the loop
    last_chAB_hist = uw_end

    #### loop through the possible permutations
    for i0, v0 in enumerate(Chs_perms2):
        
        #### get the two channels under consideration from permutation list
        chA, chB = v0[0:2]
        print('channels:', chA, ' & ', chB)

        #### generate dts from arrival time lists
        gen_dts_from_tts(d2, d3, 'FCT', chA, chB)
        
        #### log the time taken
        chAB_end = time.time()
        chAB_time = int(chAB_end - last_ch_AB_hist)
        chAB_time_str = str(datetime.timedelta(seconds=chAB_time))
        write_str = ('channels:' + chA + ' & ' + chB + '\n' +
                     'dt calc finished = ' + chAB_time_str + '\n')
        log.write(write_str)

        #### histogram the lists as well, for quick check of data
        hist_1d(d2, res, t_range, chA, chB)
        plot_1d_hist(d2, t_lim, chA, chB)
        chAB_hist_end = time.time()
        chAB_hist = int(chAB_hist - chAB_time)
        chAB_hist_str = str(datetime.timedelta(seconds=chAB_hist))
        write_str = ('channels:' + chA + ' & ' + chB + '\n' +
                     'g2 hist finished = ' + chAB_hist_str + '\n')
        log.write(write_str)

        #### assign time value to last_chAB_hist for next time log
        last_chAB_hist = ch_AB_hist


    #### generate list of possible combinations of channels defined above
    Chs_combs3 = list(set(permutations(Chs, 3)))

    #### this is for logging the timings in the loop
    last_chABAC = last_chAB_hist

    #### for each triplet combination histogram the dt pair
    #### (needs pairwaise dt calcs to be done)
    for i0, v0 in enumerate(Chs_combs3):
        chA, chB, chC = v0[0:3]
        print('channels:', chA, chB, ' & ', chA, chC)

        #### calculate usual g3 dts and plot
        hist_2d(d2, g3_res, t_range, chA, chB, chC)
        plot_2d_hist(d2, t_lim, g3_res, t_range, chA, chB, chC)

        #### calculate the 2 alternative dt methods for g3
        hist_2d_alt(d2, g3_res, t_range, chA, chB, chC)
        hist_2d_alt_abs(d2, g3_res, t_range, chA, chB, chC)

        #### log processing time
        chABAC_end = time.time()
        chABAC_hist = int(chABAC_hist - last_chABAC)
        chABAC_hist_str = str(datetime.timedelta(seconds=chABAC_hist))
        write_str = ('channels:' + chA + chB + ' & ' + chA + chC + '\n'
                     'g3 hist finished =' + chABAC_hist_str + '\n')
        log.write(write_str)
        last_chABAC = chABAC_end

    #### recond time for the whole code
    dt_hist_end = time.time()
    
    proc_time_total = int(dt_hist_end - start_time)
    proc_time_total_str =  str(datetime.timedelta(seconds=proc_time_total))
    write_str = ('Total time: ' + proc_time_total_str)

    print('Total time', proc_time_total_str)



