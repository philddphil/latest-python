# Version notes
# Focussing on the HBT times and minimising runtime

##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import os
##############################################################################
# Do some stuff
##############################################################################
# Number of photons in experiment
γs_sent = 5e7
# Time steps (ns)
Δt = 1
# Excited state lifetime (ns)
τ_decay = 10
# τ_excite means excitation is τ_excite times faster than τ_decay
τ_excite = 1
# System transmission (%)
T = 1

# initiate stuff for calcs
HBT_click = 0
delay = 0
displayed = 0

# initiate files and paths
p0 = r'D:\Experimental Data\Python simulations (G5 A5)'\
    r'\Single photon statistics\Data\20190320\\'\
    'cw ' + str(τ_decay) + 'ns, ' + \
    str(τ_excite) + 'x exc, ' + \
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
            if '# photons sent =' in j0:
                γs_prev = float(j0.split(' = ')[-1])
                print('γs sent so far =', γs_prev)
            if '# photons det 1 =' in j0:
                γs_det1 = float(j0.split(' = ')[-1])
                print('γs @ det 1 so far =', γs_det1)
            if '# photons det 2 =' in j0:
                γs_det2 = float(j0.split(' = ')[-1])
                print('γs @ det 2 so far =', γs_det2)

else:
    t_clk = 0
    γs_prev = 0
    γs_det1 = 0
    γs_det2 = 0
    print('restarted clock')
    print('restarted counts')
    with open(f5, 'w+', encoding='utf-8') as f:
        f.write('τ decay = '"%s\n" % τ_decay)
        f.write('τ excite = '"%s\n" % τ_excite)
        f.write('Δt = '"%s\n" % Δt)
        f.write('# photons sent = '"%s\n" % γs_sent)
        f.write('# photons det 1 = '"%s\n" % γs_det1)
        f.write('# photons det 2 = '"%s\n" % γs_det2)
        f.write('exp clk = '"%s\n" % t_clk)

p2 = T / 100

for i0, j0 in enumerate(np.arange(γs_sent)):
    # set excitation rate 10 times quicket than decay rate
    p0 = np.random.random()
    t0 = 1 - (τ_decay / τ_excite) * np.log(1 - p0)

    # determine decay time
    # note this limits the timesteps to integers (i.e. ns)
    p1 = np.random.random()
    t1 = - τ_decay * np.log(p1 / Δt)

    t = t1 + t0

    # decide if γ does to det 1...
    if np.random.random() < 0.5:

        # factor in system T
        if np.random.random() < p2:
            # count detection event @ det 1
            γs_det1 = γs_det1 + 1
            # HBT logic
            if HBT_click == 0:
                HBT_click = 1
                t_start = t + t_clk + delay
            else:
                HBT_click = 0
                τ_HBT = t_start - t - t_clk
                with open(f4, 'a') as f:
                    f.write("%s\n" % τ_HBT)

    # or γ goes to det 2...
    else:
        # factor in system T
        if np.random.random() < p2:
            # count detection event @ det 2
            γs_det2 = γs_det2 + 1
            # HBT logic
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
γs_sent = γs_sent + γs_prev
print('total sent rate = ',
      np.round(1e3 * (γs_sent / t_clk), 2), 'Mcps')
print('total detection rate = ',
      np.round(1e3 * ((γs_det1 + γs_det2) / t_clk), 2), 'Mcps')
print('measured T = ', np.round((γs_det1 + γs_det2) / γs_sent, 2))

with open(f5, 'w', encoding='utf-8') as f:
    f.write('τ decay = '"%s\n" % τ_decay)
    f.write('τ excite = '"%s\n" % τ_excite)
    f.write('Δt = '"%s\n" % Δt)
    f.write('# photons sent = '"%s\n" % γs_sent)
    f.write('# photons det 1 = '"%s\n" % γs_det1)
    f.write('# photons det 2 = '"%s\n" % γs_det2)
    f.write('exp clk = '"%s\n" % t_clk)
