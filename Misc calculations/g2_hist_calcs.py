# %%
import numpy as np

Error_req_percent = 0.05

Bin_pop = (1/(Error_req_percent/100))**2

print('Target Bin population:', 2 * Bin_pop)


f = 58e3
mean_photon_number = 0.01

D1_cps = f * mean_photon_number
D2_cps = D1_cps 

print( 'Detector 1 counts per second:', D1_cps, 'cps')

T = (2*Bin_pop * f) / (D1_cps * D2_cps)

print('Time to fill histogram:', T, ' sec')
print('Time to fill histogram:', T/60, ' min')
print('Time to fill histogram:', T/(60*60), ' hrs')
print('Time to fill histogram:', T/(60*60*24), ' days')

# %%
