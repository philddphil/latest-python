# Version notes
# Focussing on the HBT times and minimising runtime

##############################################################################
# Import some libraries
##############################################################################
import numpy as np

##############################################################################
# Do some stuff
##############################################################################
# Number of photons in experiment
γs = 1e5
# Time steps (ns)
Δt = 1
# Excited state lifetime (ns)
τ_decay = 1
# τ_excite means excitation is τ_excite times faster than τ_decay
τ_excite = 10
# System transmission (%)
T = 5

# initiate stuff for calcs
t_clk = 0
HBT_click = 0
delay = 0
displayed = 0

# initiate files and paths
p0 = r'D:\Experimental Data\Python simulations (G5 A5)'\
    r'\Single photon statistics\Data'

f4 = p0 + r'\\' + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    'T ' + str(round(T, 2)) + '% - HBT.txt'
f5 = p0 + r'\\' + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
    'T ' + str(round(T, 2)) + '% - fom.txt'
# read exp clk value
with open(f5, 'r', encoding='utf-8') as f:
    a = f.read()
    b = a.split('\n')
    for i0, j0 in enumerate(b):
        if 'exp clk =' in j0:
            t_clk = float(j0.split(' = ')[-1])
            print(t_clk)

p2 = T / 100

for i0, j0 in enumerate(np.arange(γs)):
    # set excitation rate 10 times quicket than decay rate
    p0 = np.random.random()
    t0 = 1 - τ_decay * τ_excite * np.log(1 - p0)

    # determine decay time
    # note this limits the timesteps to integers (i.e. ns)
    p1 = np.random.random()
    t1 = -1 * τ_decay * np.log(p1 / Δt)

    t = int(t1 + t0)

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
    with open(f5, 'w', encoding='utf-8') as f:
        f.write('τ decay = '"%s\n" % τ_decay)
        f.write('τ excite = '"%s\n" % τ_excite)
        f.write('Δt = '"%s\n" % Δt)
        f.write('# photons = '"%s\n" % γs)
        f.write('exp clk = '"%s\n" % γs)
    display, __ = divmod(t_clk, 100000)
    if display != displayed:
        print(display)
        displayed = display
