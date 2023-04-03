import numpy as np
import binarystate as bst

def rhs(H, A, w):
    A_H = np.zeros(8, dtype=np.complex128)
    for k,H_i in enumerate(H):
        for i,a in enumerate(A):
            s = bst.IndToState(i)
            s_H = np.array([H_i[i]@s[i] for i in range(3)])
            sABS = np.abs(s_H)
            T = np.array([np.dot(sABS[i],np.transpose(s_H[i])) for i in range(3)  ])
            T = np.prod(T) 
            j = bst.StateToInd(sABS)
            A_H[j] = A_H[j] + a*T*w[k]
    return -1j*A_H




