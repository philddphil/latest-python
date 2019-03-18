# Version notes
# Focussing on the HBT times and minimising runtime
# Pulsed excitaion
##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import os
##############################################################################
# Do some stuff
##############################################################################
# Number of photons in experiment
γs = 1e6
# Time steps (ns)
Δt = 1
# Excited state lifetime (ns)
τ_decay = 1
# τ_excite means excitation is τ_excite times faster than τ_decay
τ_excite = 1
# System transmission (%)
T = 10
# Pump rate (GHz)
k_12 = 0.01

# initiate stuff for calcs
HBT_click = 0
delay = 0
displayed = 0

# initiate files and paths
# Office PC (eea) path
p0 = r'D:\Experimental Data\Python simulations (G5 A5)'\
    r'\Single photon statistics\Data\pulsed ' + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    str(k_12) + 'k_12,' + \
    'T ' + str(round(T, 2)) + '%'

# Surface Pro path
p0 = r"C:\Users\Philip\Documents\Data\pulsed " + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    str(k_12) + 'k_12,' + \
    'T ' + str(round(T, 2)) + '%'

f4 = p0 + ' - HBT.txt'
f5 = p0 + ' - fom.txt'

# read exp clk value
if os.path.isfile(f5) is True:
    with open(f5, 'r', encoding='utf-8') as f:
        a = f.read()
        b = a.split('\n')
        for i0, j0 in enumerate(b):
            if 'exp clk =' in j0:
                t_clk = float(j0.split(' = ')[-1])
                print('t start = ', t_clk)
            if '# photons =' in j0:
                γs_prev = float(j0.split(' = ')[-1])
                print('γs so far =', γs_prev)
else:
    t_clk = 0
    print('restarted clock')
    γs_prev = 0
    print('restarted counts')

p2 = T / 100
t_excite = 1 / k_12
t2 = 0

for i0, j0 in enumerate(np.arange(γs)):
    # set excitation rate 10 times quicket than decay rate
    p0 = np.random.random()
    t0 = 1 - (τ_decay / τ_excite) * np.log(1 - p0)

    # determine decay time
    # note this limits the timesteps to integers (i.e. ns)
    p1 = np.random.random()
    t1 = - τ_decay * np.log(p1 / Δt)

    t = int(t2 + t1 + t0)
    q, r = divmod(t, t_excite)
    t2 = t_excite - r

    if np.random.random() < 0.5:
        if np.random.random() < p2:
            # τs1.append(t)
            if HBT_click == 0:
                HBT_click = 1
                t_start = t + t_clk + delay
            else:
                HBT_click = 0
                τ_HBT = t_start - t - t_clk
                with open(f4, 'a') as f:
                    f.write("%s\n" % τ_HBT)
    else:
        if np.random.random() < p2:
            # τs2.append(t)
            if HBT_click == 1:
                τ_HBT = t + t_clk - t_start
                with open(f4, 'a') as f:
                    f.write("%s\n" % τ_HBT)
                HBT_click = 0
            else:
                HBT_click = 1
                t_start = t + t_clk

    t_clk = t + t_clk
    # write figures-of-merit file

    display, __ = divmod(t_clk, 100000)
    if display != displayed:
        print(display)
        displayed = display
γs = γs + γs_prev
print('total count rate = ', 1e3 * (γs / t_clk), 'Mcps')
with open(f5, 'w', encoding='utf-8') as f:
    f.write('τ decay = '"%s\n" % τ_decay)
    f.write('τ excite = '"%s\n" % τ_excite)
    f.write('Δt = '"%s\n" % Δt)
    f.write('# photons = '"%s\n" % γs)
    f.write('exp clk = '"%s\n" % t_clk)
