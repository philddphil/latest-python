##############################################################################
# Import some libraries
##############################################################################
import datetime

##############################################################################
# Do some stuff
##############################################################################
R = 7.52e3
N = 1
Res = 3e-9

# Calulate times for various g(τ)s. Note factor of 4 for g3 because of the exp arrangement

G2_Time = N / (Res * R**2)
G3_Time = N / (4 * Res**2 * R**3)
G4_Time = N / (Res**3 * R**4)

print('g^2(τ) time', str(datetime.timedelta(seconds=G2_Time)))
print('g^3(τ) time', str(datetime.timedelta(seconds=G3_Time)))
print('g^4(τ) time', str(datetime.timedelta(seconds=G4_Time)))


