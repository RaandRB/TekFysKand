import numpy as np
import binarystate as bst

def rhs(H, A):
    A_H = np.zeros(8, dtype=np.complex128)
    for k,H_i in enumerate(H):
        weight = np.linalg.norm(H_i[0])/np.sqrt(2)
        H_norm = H_i/weight
        for i,a in enumerate(A):
            s = bst.IndToState(i)
            s_H = np.array([H_norm[i]@s[i] for i in range(3)])
            sABS = np.abs(s_H)
            T = np.array([np.dot(sABS[i],np.transpose(s_H[i])) for i in range(3)  ])
            T = np.prod(T) 
            j = bst.StateToInd(sABS)
            A_H[j] = A_H[j] + a*T*weight
    return -1j*A_H




