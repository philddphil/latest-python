##############################################################################
# Import some libraries
##############################################################################
import os
import csv
import glob
import numpy as np

##############################################################################
# Import data (saved by python code filter data.py)
##############################################################################
# Specify directory and datasets
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211")
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212"
      r"\g4_1MHzPQ_48dB_cont_snippet_3e6")
# d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212\
#   g4_1MHzTxPIC_55dB_cont_snippet_3e6")
d1 = d0 + r'\Py data'
os.chdir(d1)
datafiles0 = glob.glob(d1 + r'\*ch1*')
datafiles1 = glob.glob(d1 + r'\*ch2*')
datafiles2 = glob.glob(d1 + r'\*ch3*')
dt_chs = '2d_dts_chs_12_13_'
# define a ROI range to check for co-incidences over
locale = 1000
# define range of τs to be stored
t_range = 10000000
# i2 range
i3_rng = np.arange(0, 1)

global_dts = []
global_cps0 = []
global_cps1 = []
global_cps2 = []
global_t = 0

for i0, v0 in enumerate(datafiles0[0:]):

    # i0 = 2
    print('file #', i0, ' ', datafiles0[i0])
    # 1e-7 is the saved resolution - this is 0.1 microsecond
    tta = np.loadtxt(datafiles0[i0])
    ttb = np.loadtxt(datafiles1[i0])
    ttc = np.loadtxt(datafiles2[i0])

    # convert channels to ns
    # use channel 0 as the start channel
    tt0_ns = [j0 * 1e-1 for j0 in tta]
    # subsequent channels can be used as the stop channels
    tt1_ns = [j0 * 1e-1 for j0 in ttb]
    tt2_ns = [j0 * 1e-1 for j0 in ttc]

    stop_tts = [tt1_ns, tt2_ns]

    # calc total time and # counts, count rates & gradient functions
    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns), np.max(tt1_ns)]) * 1e-9
    global_t += total_t
    c0 = len(tt0_ns)
    c1 = len(tt1_ns)
    c2 = len(tt2_ns)

    cps0 = c0 / total_t
    cps1 = c1 / total_t
    cps2 = c2 / total_t

    global_cps0.append(cps0)
    global_cps1.append(cps1)
    global_cps2.append(cps2)

    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)
    dydx2 = np.max(tt2_ns) / len(tt2_ns)

    stop_dydxs = [dydx1, dydx2]

    # print some values
    print('time for file ', np.round(total_t, 5), 's')
    print('total time collected ', np.round(global_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')
    print('Ctr 3 - ', np.round(cps2 / 1000, 2), 'k counts per second')

    ##########################################################################
    # Perform start-stop measurements
    ##########################################################################

    # loop through tt0_ns as our start channel
    for i1, v1 in enumerate(tt0_ns[0:]):
        dts_i = []
        for i2, v2 in enumerate(stop_tts):
            
            tti_ns = v2
            dydxi = stop_dydxs[i2]
            # pad 'stop' channel with 0s
            tt_pad = np.pad(tti_ns, locale + 1)
            # trunkate full tt1_ns list to a region of 2 * locale values around
            # the same time as the value v1 is (need to use dydx1 to convert)

            # 1. find the corresponding index in tt1_ns
            i_tti = int(v1 / dydxi)

            # 2.  specify locale around idx to check - get both times & idx
            # vals
            tt_local = tt_pad[i_tti:i_tti + 2 * locale]
            x_local = np.arange(i_tti, i_tti + 2 * locale) - locale + 1

            # substract ith value of tt0_ns (v1) from tt1_ns & use abs operator
            tt_temp = np.abs(tt_local - v1)

            # find idx of min
            HBT_idx0 = np.argmin(tt_temp)

            # find time difference of min value
            HBT_test0 = tt_local[HBT_idx0] - v1

            # if time difference within range, analyse sub-region
            if - t_range < HBT_test0 < t_range:

                # need loop to check another sub-region, around the minimum
                # value
                for i3 in i3_rng:

                    # find idx of min
                    HBT_idx1 = HBT_idx0 + i3
                    # print(HBT_idx0, HBT_idx1)
                    # find time difference of min value
                    HBT_test1 = tt_local[HBT_idx1] - v1

                    # check if value is of interest
                    if -t_range < HBT_test1 < t_range:
                        dts_i.append(HBT_test1)

        global_dts.append(dts_i)

    (q, r) = divmod(i0, 10)
    print(r, q)
    if r == 0 and q != 0:
        print('saving dts')
        print(type(global_dts))
        print(type(global_dts[0]))
        dt_file = dt_chs + str(q - 1) + '.csv'
        with open(dt_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(global_dts)
        global_dts = []


dt_file = dt_chs + str(q - 1) + '.csv'
with open(dt_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(global_dts)

global_cps0 = np.mean(global_cps0)
global_cps1 = np.mean(global_cps1)
global_cps2 = np.mean(global_cps2)

##############################################################################
# Save global Histogram values & info
##############################################################################
np.savetxt("other_globals_g3.csv",
           [global_t, global_cps0, global_cps1, global_cps2],
           delimiter=',')
