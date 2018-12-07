import numpy as np

from glob import *

import os

import time

from scitools.std import *


for filename in glob('wave_*.png'):

    os.remove(filename)

def f(x,t):

    return exp(-(x-3*t)**2)*sin(3*pi*(x-t))


x_array = linspace(6, 6, 1001)

t_array = linspace(-1, 1, 61)

counter = 0


for t in t_array:

    plot(x_array, f(x_array, t), 'r-', axis=[-6, 6, -1.2, 1.2], xlabel='x', ylabel='f(x)', legend='t = %.2f' %(t), savefig = 'wave_%04d.png' %(counter))

    time.sleep(0.06)

    counter += 1


cmd = 'convert -delay 6 wave_*.png plot_wavepacket_movie.gif'

os.system(cmd)

raw_input('Press enter to exit')