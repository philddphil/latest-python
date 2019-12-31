##############################################################################
# Import some libraries
##############################################################################
import numpy as np

##############################################################################
# Do some stuff
##############################################################################
# Number of photons in experiment
γs = 1e6
# Time steps (ns)
Δt = 1
# Excited state lifetime (ns)
τ = 10
# System transmission (%)
T = 5

# initiate stuff for calcs
ts_rel = []
ts_abs = []
ts1 = []
ts2 = []
ts_HBT = []
t_clk = 0
HBT_click = 0
delay = 0

p2 = T / 100

for i0, j0 in enumerate(np.arange(γs)):
    # set excitation rate 10 times quicket than decay rate
    p0 = np.random.random()
    t0 = 1 - 2 * τ * np.log(1 - p0)
    # t0 = 0
    # determine decay time
    # note this limits the timesteps to integers (i.e. ns)
    p1 = np.random.random()
    t1 = -1 * τ * np.log(p1 / Δt)

    t = t1 + t0
    ts_rel.append(t)
    ts_abs.append(t + t_clk)

    if np.random.random() < 0.5:
        if np.random.random() < p2:
            ts1.append(t + t_clk)
            if HBT_click == 0:
                HBT_click = 1
                t_start = t + t_clk + delay
            else:
                HBT_click = 0
                ts_HBT.append(t_start - t - t_clk)
    else:
        if np.random.random() < p2:
            ts2.append(t + t_clk)
            if HBT_click == 1:
                ts_HBT.append(t + t_clk - t_start)
                HBT_click = 0
            else:
                HBT_click = 1
                t_start = t + t_clk

    t_clk = t + t_clk

# saves lists of times
p0 = r'C:\local files\Experimental Data\G5 A5 Python simulations'\
    r'\Single photon statistics\Data\20191211'

f0 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' ts abs.txt'
f1 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' ts rel.txt'
f2 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' ts 1.txt'
f3 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' ts 2.txt'
f4 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' ts HBT.txt'
f5 = p0 + r'\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' fom.txt'

with open(f0, 'w') as f:
    for item in ts_abs:
        f.write("%s\n" % item)

with open(f1, 'w') as f:
    for item in ts_rel:
        f.write("%s\n" % item)

with open(f2, 'w') as f:
    for item in ts1:
        f.write("%s\n" % item)

with open(f3, 'w') as f:
    for item in ts2:
        f.write("%s\n" % item)

with open(f4, 'w') as f:
    for item in ts_HBT:
        f.write("%s\n" % item)

with open(f5, 'w') as f:
    f.write("%s\n" % τ)
    f.write("%s\n" % Δt)
    f.write("%s\n" % γs)
