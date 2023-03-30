import numpy as np
import binarystate as bst

def rhs(H, A):
    A_H = np.zeros(8, dtype=np.complex128)
    for H_i in H:
        for i,a in enumerate(A):
            s = bst.IndToState(i)
            s_H = np.array([H_i[i]@s[i] for i in range(3)])
            sABS = np.abs(s_H)
            T = np.array([np.dot(sABS[i],np.transpose(s_H[i])) for i in range(3)  ])
            T = np.prod(T) 
            j = bst.StateToInd(sABS)
            A_H[j] = A_H[j] + a*T
    return -1j*A_H




