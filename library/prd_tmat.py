##############################################################################
# Import some libraries
##############################################################################
import os
import glob
import copy
import random
import datetime

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

from scipy import ndimage
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from PIL import Image


###############################################################################
# ABCD matrix defs
###############################################################################
def ABCD_d(q_in, d, n=1):
    M = np.array([[1, d * n], [0, 1]])
    q_out = np.matmul(M, q_in)
    return(q_out)


def ABCD_propagate(q0, z_end, z_start=0, res=1000, n=1):
    qz = [q0]
    zs = np.linspace(z_start, z_end, res)
    ns = n * np.ones(len(zs))
    if q0[1] == 1:
        z_start = np.real(q0[0])

    dz = zs[1] - zs[0]

    for i1, val1 in enumerate(zs[1:]):
        q1 = ABCD_d(q0, dz, n)
        qz.append(q1)
        q0 = q1

    return(zs, qz, ns)


def ABCD_tlens(q_in, f):
    M = np.array([[1, 0], [-1 / f, 1]])
    q_out = np.matmul(M, q_in)
    if q_in[1] == 1:
        q_out = q_out / q_out[1]
    return(q_out)


def ABCD_plan(q_in, n1, n2):
    M = np.array([[1, 0], [0, n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) is True:
        q_out = q_out / q_out[1]
    return(q_out)


def ABCD_curv(q_in, n1, n2, R):
    M = np.array([[1, 0], [(n1 - n2) / (R * n2), n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) is True:
        q_out = q_out / q_out[1]
    return(q_out)
