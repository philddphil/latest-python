##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\Pressure measurements (F5 L10)"
      r"\Pressure 20181218\Time series 1.txt")
p1 = (r"D:\Experimental Data\Pressure measurements (F5 L10)"
      r"\Pressure 20181219\Time series 2.txt")
p2 = (r"D:\Experimental Data\Pressure measurements (F5 L10)"
      r"\Pressure 20181219\Time series 3.txt")

Δts0, Ps0 = prd.load_Pressure(p0)
Δts1, Ps1 = prd.load_Pressure(p1)
Δts2, Ps2 = prd.load_Pressure(p2)
##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()

###

fig1 = plt.figure('fig1', figsize=(10, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('time (min)')
ax1.set_ylabel('pressure (mbar)')
ax1.set_yscale('log')
plt.plot(Δts0, Ps0, 'o--', label='Original')
plt.plot(Δts1, Ps1, 'o--', label='1st clean')
plt.plot(Δts2, Ps2, 'o--', label='2nd clean')
plt.tight_layout()

ax1.legend(loc='upper left', fancybox=True, framealpha=0.5)
plt.show()
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
