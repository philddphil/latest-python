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
p1_no = 15
p1_date = '2018-07-18'
p1_ATVI = pdr.get_data_yahoo('ATVI', p1_date, p1_date)
p1_GBP = pdr.get_data_fred('DEXUSUK', p1_date, p1_date)
print(p1_GBP)
p1_cost_USD = p1_no * float(p1_ATVI['Open'])
p1_cost_GBP = p1_no * float(p1_ATVI['Open']) / 1.2843
print('Amount paid = $', p1_cost_USD, ' = £ ', p1_cost_GBP)

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(ATVI_short['Close'],'.:')
# plt.show()
