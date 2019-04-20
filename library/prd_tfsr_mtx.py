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