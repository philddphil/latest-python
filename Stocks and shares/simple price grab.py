import pandas_datareader as pdr
from datetime import datetime
print('starting to grab data')
ATVI = pdr.get_data_yahoo('ATVI')
print(ATVI)
