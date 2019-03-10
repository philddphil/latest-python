import matplotlib.pyplot as plt
import numpy as np
import copy


# Define functions
def tfm(n1, n2):	 # transfer matrix for passing fields from n1 to n2
    a = (n1 + n2) / (2 * n2)
    b = (n2 - n1) / (2 * n2)
    M = np.array([[a, b], [b, a]])
    return(M)


def dx(x, k):		 # propagate over distance x
    Dx = np.array([[np.exp(-1j * x * k)], [np.exp(1j * x * k)]])
    return(Dx)


# Set the operating wavelength
wl = 1.55e-6
L = 0.8
N = 30000
# Build some arrays
wls = np.linspace(wl - 10e-12, wl + 10e-12, N)
Rs = np.zeros((1, N))

# Set some refractive indices
n0 = 12.3  									# nr to get R = 85%
n1 = 1     									# nr of air
n2 = np.sqrt(1.44) 							# nr of MgF2
n3 = 1.44  									# nr of SiO2
n4 = 1  									# nr of oil

for i1 in np.arange(N):

    # Calc associated k vectors
    k0 = (2 * np.pi * n0) / wls[i1] 		# n1 = 12.3 (mirror)
    k1 = (2 * np.pi * n1) / wls[i1] 		# n1 = 1 (air)
    k2 = (2 * np.pi * n2) / wls[i1] 		# n2 = 1.38 (MgF2)
    k3 = (2 * np.pi * n3) / wls[i1] 		# n3 = 1.44 (SiO2)
    k4 = (2 * np.pi * n4) / wls[i1] 		# n4 = 1.58 (Oil)

    # Use transfer matrix to pass field through structure
    E0 = np.array([[1], [0]])

    E1 = np.dot(tfm(n0, n1), E0)			# LCOS and WSS
    E2 = dx(L, k1) * E1

    E3 = np.dot(tfm(n1, n2), E2)			# Lenslet AR side 1
    E4 = dx(wl / (4 * n2), k2) * E3

    E5 = np.dot(tfm(n2, n3), E4)			# Lenslet substrate
    E6 = dx(1e-3, k3) * E5

    E7 = np.dot(tfm(n3, n2), E6)			# Lenslet AR side 2
    E8 = dx(wl / (4 * n2), k2) * E7

    E9 = np.dot(tfm(n2, n4), E8)			# Gap between lenslet & fibre
    E10 = dx(10e-6 + 0.0*wl, k4) * E9

    E11 = np.dot(tfm(n4, n2), E10)
    E12 = dx(wl / (4 * n2), k2) * E11

    E13 = np.dot(tfm(n2, n3), E12)			# Fibre out

    EF = copy.copy(E13)
    Ii = abs(EF[0])**2
    Ir = abs(EF[1])**2
    R = Ir / Ii
    Rs[0, [i1]] = copy.copy(R)

wls = np.squeeze(copy.copy(wls)) - wl
Rs = np.squeeze(copy.copy(Rs))
Ts = 1 - Rs
Noise = np.max(Rs) - np.min(Rs)
plt.figure('λ response')
plt.plot(wls * 1e12, Rs, 'r')
plt.plot(wls * 1e12, Ts, 'b')
plt.ylim(0, 1)
plt.xlabel('∆ Wavelength, (λ) pm')
plt.ylabel('R or T')
plt.title('Max - min = ''%.2f' % Noise)


plt.savefig('demo.png', transparent=True)
plt.show()
print(Noise)