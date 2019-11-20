##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import copy


###############################################################################
# ABCD matrix defs
###############################################################################
# Follows the wikipedia ray matrix formalism

# q parameter is defined as 1/q = 1/R - i*λ0/π*n*(w**2)
# p is a normal ray trace

# This is the matrix multiplication for propagation
# takes the input vector (q_in) and multiplies it by the matrix corresponding
# to a length of d (optical path length = n * d)
def ABCD_MM(q_in, d, n=1):
    M = np.array([[1, d * n], [0, 1]])
    q_out = np.matmul(M, q_in)
    return(q_out)

# propagation from z_in to z_end with 1000 pts in between. ###################
def ABCD_propagate(qs, z_end, zs_in=None, ns_in=None, res=1000):
    if zs_in is None:
        zs_in = [0]
    if ns_in is None:
        ns_in = [1]
    zs_out = copy.copy(zs_in)
    qz = qs
    q0 = qs[-1]
    z_start = zs_in[-1]
    zs_i = np.linspace(z_start, z_end, res)
    ns = ns_in[-1] * np.ones(len(zs_i))
    ns_out = copy.copy(ns_in)
    if q0[1] == 1:
        z_start = np.real(q0[0])

    dz = zs_i[1] - zs_i[0]

    for i1, val1 in enumerate(zs_i[0:]):
        q1 = ABCD_MM(q0, dz, ns[i1])
        qz.append(q1)
        q0 = q1
        zs_out.append(zs_i[i1])
        ns_out.append(ns[i1])

    return(zs_out, qz, ns_out)

# refraction due to a thin lens ##############################################
def ABCD_tlens(qs, f):
    M = np.array([[1, 0], [-1 / f, 1]])
    q_out = np.matmul(M, qs[-1])
    if qs[-1][1] == 1:
        q_out = q_out / q_out[1]
    qs[-1] = q_out
    return qs

# refraction at a planar interface ###########################################
def ABCD_plan(q_in, n1, n2):
    M = np.array([[1, 0], [0, n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) is True:
        q_out = q_out / q_out[1]
    return(q_out)

# refraction at a curved interface ###########################################
def ABCD_curv(q_in, n1, n2, R):
    M = np.array([[1, 0], [(n1 - n2) / (R * n2), n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) is True:
        q_out = q_out / q_out[1]
    return(q_out)
