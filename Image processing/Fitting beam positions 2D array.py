#  Fitting_beam_positions.py
#
#  Code to analyse input/output array alignment
#
#  ROADMap Systems Ltd
#
#  Brian Robertson
#
#  20/November/2017
#
#  Version 1 - Basic code. (20/11/2017)


from numpy import*
from pylab import *
from math import*
from scipy.optimize import minimize

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

# Define functions

# Calculate theoretical beam positions
def calculate(scale, angle, ex, ey, x, y):

    error = 0

    for ii in range(3):
        ii2 = ii - 1
        for jj in range(3):
            jj2 = jj - 1

            posx = x[ii, jj]
            posy = y[ii, jj]

            posx_ideal = ii2 * scale * \
                np.cos(angle) + jj2 * scale * np.sin(angle)
            posy_ideal = ii2 * scale * \
                np.sin(angle) - jj2 * scale * np.cos(angle)

            x_theory[ii, jj] = posx_ideal + ex
            y_theory[ii, jj] = posy_ideal + ey

            error_new = (posx - posx_ideal - ex)**2 + \
                (posy - posy_ideal - ey)**2
            error = error + error_new

    return x_theory, y_theory, error

# Minimization function
def fun(x00, x, y):

    scale = x00[0]
    angle = x00[1]
    ex = x00[2]
    ey = x00[3]

    error = 0

    for ii in range(3):
        ii2 = ii - 1
        for jj in range(3):
            jj2 = jj - 1

            posx = x[ii, jj]
            posy = y[ii, jj]

            posx_ideal = ii2 * scale * \
                np.cos(angle) + jj2 * scale * np.sin(angle)
            posy_ideal = ii2 * scale * \
                np.sin(angle) - jj2 * scale * np.cos(angle)

            x_theory[ii, jj] = posx_ideal + ex
            y_theory[ii, jj] = posy_ideal + ey

            error_new = (posx - posx_ideal - ex)**2 + \
                (posy - posy_ideal - ey)**2
            error = error + error_new

    return error

# Main code

# Experimental data

x0 = np.array([1.407, 1.436, 1.459, 1.739, 1.784, 1.798, 2.076, 2.111, 2.103])
x0m = np.reshape(x0, (3, 3)) - x0[4]
y0 = np.array([3.211, 2.820, 2.503, 3.223, 2.860, 2.513, 3.213, 2.858, 2.530])
y0m = np.reshape(y0, (3, 3)) - y0[4]

x1 = np.array([1.183, 1.220, 1.224, 1.509, 1.561, 1.575, 1.837, 1.866, 1.873])
x1m = np.reshape(x1, (3, 3)) - x1[4]
y1 = np.array([3.523, 3.137, 2.824, 3.531, 3.181, 2.839, 3.515, 3.168, 2.852])
y1m = np.reshape(y1, (3, 3)) - y1[4]

x2 = np.array([0.793, 0.831, 0.857, 1.334, 1.394, 1.410, 1.883, 1.934, 1.926])
x2m = np.reshape(x2, (3, 3)) - x2[4]
y2 = np.array([3.999, 3.373, 2.836, 4.017, 3.427, 2.854, 3.989, 3.414, 2.872])
y2m = np.reshape(y2, (3, 3)) - y2[4]

x3 = np.array([4.081, 4.129, 4.131, 4.657, 4.717, 4.667, 5.218, 5.234, 5.274])
x3m = np.reshape(x3, (3, 3)) - x3[4]
y3 = np.array([4.023, 3.446, 2.868, 4.040, 3.450, 2.799, 4.054, 3.477, 2.920])
y3m = np.reshape(y3, (3, 3)) - y3[4]


x = x3m
y = y3m

# Define theoretical position matrix
x_theory = np.zeros([3, 3])
y_theory = np.zeros([3, 3])

# Initial parameters (best guess)
scale = 0.35
angle = 2 * pi / 180
ex = 0
ey = 0

# Set up parameters for minimization function
x00 = [scale, angle, ex, ey]
# Optimization using function 'fun'
res = minimize(fun, x00, (x, y), method='Powell')
# Optimized parameters
print(res.x)

scale = res.x[0]
angle = res.x[1]
ex = res.x[2]
ey = res.x[3]

print('Scale =', np.round(scale, 3))
print('Angle =', np.round(angle * 180 / pi, 2))
print('ex =', np.round(ex, 3))
print('ey =', np.round(ey, 3))

# Calculate theoretical positions
x_theory, y_theory, error = calculate(scale, angle, ex, ey, x, y)
print('error = ', np.round(error, 3))

# Plot data

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.plot(x, y, 'o')
plt.plot(x_theory, y_theory, '+')

show()

prd.PPT_save_2d(fig1, ax1, '2D array fit 1')
