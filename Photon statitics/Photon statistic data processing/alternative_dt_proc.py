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
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################################################
# Some defs
##############################################################################
# Call proc_n_lines to process a .lst file into four #py.txt arrival times ###
def proc_lst(d0, array_limit=100):
    os.chdir(d0)

    d1 = d0 + r'\dts'

    try:
        os.mkdir(d1)
    except OSError:
        print("Creation of the directory %s failed" % d1)
    else:
        print("Successfully created the directory %s " % d1)


    f0 = d0 + r"\TEST.lst"
    data_start = 1000
    scale = 16
    
    t0, t1, missed_counts, data_line = 0, 0, 0, 0
    array_size, array_number, current_line, tot_times = 0, 0, 0, 0
    data_array, data_row = [], []
    
    with open('photonlog.txt', 'w+') as log:

        with open(f0, 'r') as input_file:

            for line in input_file:
                if '[DATA]' in line:
                    data_start = current_line
                    print('data parsing...')

                if current_line > data_start:
                    line_hex = line.rstrip('\n')
                    
                    line_int = int(line_hex, scale)
                    line_bin = bin(line_int).zfill(8)
                    t_bin = line_bin[0:-4]
                    t_int = int(t_bin, 2)

                    t0 = t1
                    t1 = t_int

                    if data_line > 1:
                        dt = t1 - t0

                        if dt >= 0:
                            data_row.append(dt)

                        if dt < -1e6:
                            print('Clock reset at', current_line)
                            print('Dts in reset', len(data_row))
                            print('Array size', array_size)
                            print('Data array', len(data_array))
                            data_array.append(data_row)
                            tot_times += len(data_row)
                            print(array_size)
                            array_size = len(data_array)
                            print(array_size)
                            data_row = []

                    if array_size == array_limit:
                        fname = d1 + r'\dt_array_' + str(array_number)
                        np.save(fname, data_array)
                        array_number =+ 1
                        print('saving', fname,
                              'phtons arrivals so far', curr_line,
                              '               ',
                              'resets so far', array_number * len(data_array))
                        data_array = []
                        array_size = 0
                        write_str = ('array #:' + str(array_number) + '\t' +
                                     'dts:' + str(data_line) + '\t')
                        log.write(write_str)


                    data_line += 1
                current_line += 1


##############################################################################
# Code to do stuff
##############################################################################
# specify data directory
d0 = (r"D:\pd10\TEST4 2hr")
os.chdir(d0)

# create log file to write to

with open('log.txt', 'w+') as log:
    # log start
    start_time = time.time()
    # prepare additional directories for processed data

    # call proc_lst
    print('processing lst file into channel arrival time asciis')
    proc_lst(d0)

    # log lst end
    lst_end = time.time()
    write_str = ('lst processing end = ' +
                 str(np.round(lst_end - start_time, 3)) + '\n')
    log.write(write_str)
    print("lst proc", lst_end - start_time)
