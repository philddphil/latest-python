##############################################################################
# Import some libraries
##############################################################################
import os
import glob
import numpy as np

##############################################################################
# Import data (saved by labVIEW code controlling HH400)
##############################################################################
d0 = r"C:\Users\pd10\Documents\HH Data to process\20200310 125019"
p0 = d0 + r"\0 tt ch0.txt"
p1 = d0 + r"\1 tt ch1.txt"
os.chdir(d0)
datafiles0 = glob.glob(d0 + r'\*0.txt')
datafiles1 = glob.glob(d0 + r'\*1.txt')

# define a ROI range to check for co-incidences over
locale = 2000
# define range of τs to be stored
t_range = 100000
# i2 range
i2_rng = np.arange(0, 1)

global_dts = []
global_cps0 = []
global_cps1 = []
global_t = 0

for i0, v0 in enumerate(datafiles0[0:]):

    # i0 = 2
    print('file #', i0, ' ', datafiles0[i0])
    # 1e-7 is the saved resolution - this is 0.1 microsecond
    tta = np.loadtxt(datafiles0[i0])
    ttb = np.loadtxt(datafiles1[i0])

    # convert to ns
    tt0_ns = [j0 * 1e2 for j0 in tta]
    tt1_ns = [j0 * 1e2 for j0 in ttb]

    # calc total time and # counts, count rates & gradient functions
    total_t = np.max([np.max(tt0_ns), np.max(tt1_ns)]) * 1e-9
    global_t += total_t
    c0 = len(tt0_ns)
    c1 = len(tt1_ns)

    cps0 = c0 / total_t
    cps1 = c1 / total_t

    global_cps0.append(cps0)
    global_cps1.append(cps1)

    dydx0 = np.max(tt0_ns) / len(tt0_ns)
    dydx1 = np.max(tt1_ns) / len(tt1_ns)

    # print some values
    print('total time collected ', np.round(total_t, 5), 's')
    print('Ctr 1 - ', np.round(cps0 / 1000, 2), 'k counts per second')
    print('Ctr 2 - ', np.round(cps1 / 1000, 2), 'k counts per second')

    ##########################################################################
    # Perform start-stop measurements
    ##########################################################################

    # pad 'stop' channel with 0s
    tt_pad = np.pad(tt1_ns, locale + 1)

    # loop through tt0_ns as our start channel
    for i1, v1 in enumerate(tt0_ns[0:]):

        # trunkate full tt1_ns list to a region of 2 * locale values around
        # the same time as the value v1 is (need to use dydx1 to convert)

        # 1. find the corresponding index in tt1_ns
        i_tt1 = int(v1 / dydx1)

        # 2.  specify locale around idx to check - get both times & idx vals
        tt_local = tt_pad[i_tt1:i_tt1 + 2 * locale]
        x_local = np.arange(i_tt1, i_tt1 + 2 * locale) - locale + 1

        # substract ith value of tt0_ns (v1) from tt1_ns & use abs operator
        tt_temp = np.abs(tt_local - v1)

        # find idx of min
        HBT_idx0 = np.argmin(tt_temp)

        # find time difference of min value
        HBT_test0 = tt_local[HBT_idx0] - v1

        # if time difference within range, analyse sub-region
        if - t_range < HBT_test0 < t_range:

            # need loop to check another sub-region, around the minimum value
            for i2 in i2_rng:

                # find idx of min
                HBT_idx1 = HBT_idx0 + i2
                # print(HBT_idx0, HBT_idx1)
                # find time difference of min value
                HBT_test1 = tt_local[HBT_idx1] - v1

                # check if value is of interest
                if -t_range < HBT_test1 < t_range:
                    global_dts.append(HBT_test1)

    print(np.shape(global_dts))

    (q, r) = divmod(i0, 10)
    print(r, q)
    if r == 0 and q != 0:
        print('saving dts')
        dt_file = 'dts_' + str(q - 1) + '.csv'
        np.savetxt(dt_file, global_dts, delimiter=",")
        global_dts = []


print('saving final dts')
dt_file = 'dts_' + str(q - 1) + '.csv'
np.savetxt(dt_file, global_dts, delimiter=",")
global_dts = []


global_cps0 = np.mean(global_cps0)
global_cps1 = np.mean(global_cps1)

##############################################################################
# Save global Histogram values & info
##############################################################################
np.savetxt("other_global.csv", [global_t, global_cps0, global_cps1],
           delimiter=',')
