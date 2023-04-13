import numpy as np


def rhs(H, psi):
    return -1j*np.resize(H@psi,8)
