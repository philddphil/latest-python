##############################################################################
# Import some libraries
##############################################################################
import os
import numpy as np


##############################################################################
# Some defs
##############################################################################
def check_delay(DATAS,i0,i1):
dts = zeros(length(DATAS{i0}{i1}),1);
clk_tmp = DATAS{5}{i1};
for j0 = 1:length(DATAS{i0}{i1})
    ch_t = DATAS{i0}{i1}(j0);
    [dt, ~] = find_tick(ch_t, clk_tmp);
    dts(j0) = dt;
end
histmin = 0;
histmax = 1000;
histn = 1000;
edges = linspace(histmin,histmax,histn);
[N,edges] = histcounts(dts,edges);
[~,I] = max(N);
delay = round(edges(I));
window = [delay - 15, delay + 15];
end

##############################################################################
# Import data (saved by labVIEW code controlling HH400)
##############################################################################
# Specify directory and datasets
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200211")
d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212"
r"\g4_1MHzPQ_48dB_cont_snippet_3e6")
# d0 = (r"C:\local files\Experimental Data\F5 L9 SNSPD Fastcom tech\20200212\
#   g4_1MHzTxPIC_55dB_cont_snippet_3e6")
d1 = d0 + r'\Py data'
os.chdir(d1)
datafiles0 = glob.glob(d1 + r'\*ch0*')
datafiles1 = glob.glob(d1 + r'\*ch1*')
##############################################################################
# Do some stuff
##############################################################################
# Specify directory and datasets