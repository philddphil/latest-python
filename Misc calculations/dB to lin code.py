# %%
import numpy as np
# %%
dB = -1
lin_from_dB = 10**(dB / 10)

Pin = 1
Pout = 0.
dB_from_lin = 10 * np.log10(Pout/Pin)

print('Linear ratio from dB value:', lin_from_dB)
print('dB value from linear ratio:', dB_from_lin)
