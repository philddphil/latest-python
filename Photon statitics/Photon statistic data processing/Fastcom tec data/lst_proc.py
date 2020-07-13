##############################################################################
# Import some libraries
##############################################################################
import os
from itertools import islice


##############################################################################
# Some defs
##############################################################################

# Called by proc lst to decode a large hex file ##############################
def proc_n_lines(next_n_lines):
    scale = 16
    f1 = open('1py.txt', 'a')
    f2 = open('2py.txt', 'a')
    f3 = open('3py.txt', 'a')
    f4 = open('4py.txt', 'a')
    for i0, v0 in enumerate(next_n_lines):
        missed_counts = 0
        line_hex = v0.rstrip('\n')
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
            print(missed_counts, '@', i0, 'hex ', line_hex)
            pass

# Call proc_n_lines to process a .lst file into four #py.txt arrival times ###
def proc_lst(d0):
    os.chdir(d0)
    f0 = d0 + r"\TEST.lst"

    tot_lines = sum(1 for line in open(f0))
    print(tot_lines)
    # create files to write to
    f1 = open('1py.txt', 'w')
    f1.close()
    f2 = open('2py.txt', 'w')
    f2.close()
    f3 = open('3py.txt', 'w')
    f3.close()
    f4 = open('4py.txt', 'w')
    f4.close()

    # reopen files for appending


    lines_per_slice = 10000
    scale = 16
    with open(f0, 'r') as input_file:
        current_line = 1
        for line in input_file:
            if '[DATA]' in line:
                DATA_start = current_line
                current_slice = 0
                print(current_slice)
                while current_slice + 1 < tot_lines / lines_per_slice:
                    next_n_lines = list(islice(input_file, lines_per_slice))
                    proc_n_lines(next_n_lines)
                    current_slice += 1
                    print('slice', current_slice, 'of',
                          int(tot_lines / lines_per_slice))
            current_line += 1
##############################################################################
# Do some stuff
##############################################################################

d0 = r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200710\2"
proc_lst(d0)
