import numpy as np


def rhs(H, psi):
    return -1j/2*np.resize(H@psi,8)
