# 0. imports
import numpy as np
# 1. defs
# 2. stuff
L1 = 360e-6
L2 = 360e-6+1e-9
q = 523
λ1 = (2*L1)/q
λ2 = (2*L2)/q
Δλ = λ1 - λ2
print(np.round(Δλ*1e12), ' pm')