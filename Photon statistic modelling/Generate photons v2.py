##############################################################################
# Import some libraries
##############################################################################
import numpy as np

##############################################################################
# Do some stuff
##############################################################################
# Number of photons in experiment
γs = 1e7
# Time steps (ns)
Δt = 1
# Excited state lifetime (ns)
τ = 10
# System transmission (%)
T = 1

# initiate stuff for calcs
τs_rel = []
τs_abs = []
τs1 = []
τs2 = []
τs_HBT = []
t_clk = 0
HBT_click = 0
delay = 0

p2 = T / 100

for i0, j0 in enumerate(np.arange(γs)):
    # set excitation rate 10 times quicket than decay rate
    p0 = np.random.random()
    t0 = 1 - 2 * τ * np.log(1 - p0)

    # determine decay time
    # note this limits the timesteps to integers (i.e. ns)
    p1 = np.random.random()
    t1 = -1 * τ * np.log(p1 / Δt)

    t = int(t1 + t0)
    # τs_rel.append(t)
    # τs_abs.append(t + t_clk)

    if np.random.random() < 0.5:
        if np.random.random() < p2:
            # τs1.append(t)
            if HBT_click == 0:
                HBT_click = 1
                t_start = t + t_clk + delay
            else:
                HBT_click = 0
                τs_HBT.append(t_start - t - t_clk)
    else:
        if np.random.random() < p2:
            # τs2.append(t)
            if HBT_click == 1:
                τs_HBT.append(t + t_clk - t_start)
                HBT_click = 0
            else:
                HBT_click = 1
                t_start = t + t_clk

    t_clk = t + t_clk

# print('absolute times = ', len(τs_abs))
# print('relative times = ', len(τs_rel))
# print('d1 detections = ', len(τs1))
# print('d2 detections = ', len(τs2))
print('HBT list = ', len(τs_HBT))
# saves lists of times
f0 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' taus abs.txt'
f1 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' taus rel.txt'
f2 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' taus 1.txt'
f3 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' taus 2.txt'
f4 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' taus HBT.txt'
f5 = r'Data\1e' + str(int(np.log10(γs))) + ' Photons, ' + \
    'T = ' + str(int(T * 100)) + ' fom.txt'
# with open(f0, 'w') as f:
#     for item in τs_abs:
#         f.write("%s\n" % item)

# with open(f1, 'w') as f:
#     for item in τs_rel:
#         f.write("%s\n" % item)

# with open(f2, 'w') as f:
#     for item in τs1:
#         f.write("%s\n" % item)

# with open(f3, 'w') as f:
#     for item in τs2:
#         f.write("%s\n" % item)

with open(f4, 'w') as f:
    for item in τs_HBT:
        f.write("%s\n" % item)

with open(f5, 'w') as f:
    f.write("%s\n" % τ)
    f.write("%s\n" % Δt)
    f.write("%s\n" % γs)
