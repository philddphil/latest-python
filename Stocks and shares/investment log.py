##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from datetime import date, timedelta


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\GitHub\latest-python\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
path = (r"C:\Users\Philip\Documents\GitHub\latest-python\Stocks and shares"
        r"\investment log.txt")
file = open(path, 'r', encoding='utf-8')
data = file.readlines()
file.close()
total_paid = 0
yesterday = date.today() - timedelta(2)
print (yesterday.strftime('%d-%m-%Y'))
for i0, val in enumerate(data[1:]):
    share = str(val.split("\t")[0])
    amt = float(val.split("\t")[1])
    paid = float(val.split("\t")[2])
    date = str(val.split("\t")[3])

    current_USDGBP = pdr.get_data_fred('DEXUSUK', yesterday, yesterday)
    current_share = pdr.get_data_yahoo(share, yesterday, yesterday)
    print(current_share.loc[yesterday])
    current_price_USD = float(current_share[0])
    total_paid = total_paid + paid
    print(current_USDGBP)
# p1_ATVI = pdr.get_data_yahoo('ATVI', p1_date, p1_date)
# p1_GBP = pdr.get_data_fred('DEXUSUK', p1_date, p1_date)
# print(p1_GBP)
# p1_cost_USD = p1_no * float(p1_ATVI['Open'])
# p1_cost_GBP = p1_no * float(p1_ATVI['Open']) / 1.2843
# print('Amount paid = $', p1_cost_USD, ' = £ ', p1_cost_GBP)

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.plot(ATVI_short['Close'],'.:')
# plt.show()
