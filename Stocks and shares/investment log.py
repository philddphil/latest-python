##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import smtplib
from email.mime.text import MIMEText
from datetime import date, timedelta
from datetime import datetime
from forex_python.converter import get_rate


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"C:\Users\Phil\Documents\GitHub\latest-python\library")
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
path = (r"C:\Users\Phil\Documents\GitHub\latest-python\Stocks and shares"
        r"\investment log.txt")
file = open(path, 'r', encoding='utf-8')
data = file.readlines()

yesterday = date.today() - timedelta(3)
total_paid = 0
total_current_worth = 0
current_USDGBP = float(get_rate("USD", "GBP", yesterday))

for i0, val in enumerate(data[1:]):
    share = str(val.split("\t")[0])
    amt = float(val.split("\t")[1])
    paid = float(val.split("\t")[2])
    date = str(val.split("\t")[3])

    current_share = pdr.get_data_yahoo(share, yesterday, yesterday)
    current_price_USD = float(current_share['Adj Close'])
    current_price_GBP = current_price_USD * current_USDGBP
    current_worth = current_price_GBP * amt
    total_current_worth = current_worth + total_current_worth
    total_paid = total_paid + paid

print('Amount paid = £', total_paid)
print('Current worth = £', np.round(total_current_worth, 2))
print('Profit = £', np.round(total_current_worth - total_paid, 2))
print(r'% Gain = ', np.round((total_current_worth / total_paid - 1) * 100, 2))

s1 = ('Amount paid = £ ' + str(total_paid))
s2 = ('Current worth = £ ' + str(np.round(total_current_worth, 2)))
s3 = ('Profit = £ ' + str(np.round(total_current_worth - total_paid, 2)))
s4 = (r'% Gain = ' +
      str(np.round((total_current_worth / total_paid - 1) * 100, 2)))
email_txt = "testfile.txt"
file = open(email_txt, "w")

file.write(s1)
file.write(s2)
file.write(s3)
file.write(s4)
file.close()
# Open a plain text file for reading.  For this example, assume that

