##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
# Some defs
##############################################################################
# Find the elements immediately after the clock reset occurs #################
def find_clk_resets(a):
    b = []
    for i0, v0 in enumerate(a):
        if v0 < -10000:
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


# Call several above files to unwrap clock reset point in t data #############
# Each reset is saved in an individual .tct, so many files are saved #########
def unwrap_4ch_data(d0):
  d1 = d0 + r"\Py data"
  d2 = d1 + r"\arrival time files"
  f0 = d0 + r"\TEST.lst"

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

  f1 = d0 + r'\1py.txt'
  f2 = d0 + r'\2py.txt'
  f3 = d0 + r'\3py.txt'
  f4 = d0 + r'\4py.txt'

  os.chdir(d0)

  fs = [f1, f2, f3, f4]
  chns = 4
  print('# of channels', chns)

  # Loop over data sets, unwrap around clock resets
  # initialise list of data to unwrap
  DATAS = []

  # loop through data set
  for i0, v0 in enumerate(fs):
      # import channel data
      data = np.genfromtxt(v0)
      # differentiate to get reset points
      ddata = np.diff(data)
      # locate clock reset values
      resets = find_clk_resets(ddata)
      cycles = len(resets)
      print('unwrapping', cycles, 'cycles in ch', i0)
      # unwrap data from n * various length vectors to a list of lists
      DATAS.append(unwrap_data(data, resets))

  # Save datasets
  os.chdir(d2)
  for i0, v0 in enumerate(DATAS):
      for i1, v1 in enumerate(DATAS[i0]):
          fname = 'ch' + str(i0) + ' data' + str(i1)
          print('saving', fname)
          np.savetxt(fname, DATAS[i0][i1])


##############################################################################
# Do some stuff
##############################################################################
# Specify directory and datasets
d0 = r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200710"
 
unwrap_4ch_data(d0)