##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r'C:\Users\Phil\Documents\GitHub\python-resources\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
from datetime import datetime
print('starting to grab data')
ATVI = pdr.get_data_yahoo('ATVI')
print(ATVI['2018-07-18':'2018-07-18'])


# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(ATVI_short['Close'],'.:')
# plt.show()
	