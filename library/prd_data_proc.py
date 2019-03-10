##############################################################################
# Import some libraries
##############################################################################
import copy
import numpy as np


###############################################################################
# Data processing defs
###############################################################################
# Cosmic ray removal ##########################################################
def cos_ray_rem(data, thres):
    data_proc = copy.copy(data)
    grad_data = np.pad(np.gradient(data), 3, 'minimum')
    for n0, m0 in enumerate(grad_data):
        if m0 > thres:
            # if gradient of data[n] is above threshold, relace it with mean
            # of data[n-2] & data[n+2]
            data_proc[n0 - 2] = np.mean([data[n0 - 4], data[n0]])
    return data_proc
