import numpy as np


def StateToInd(S):
    indlist = np.array([np.dot([1,0],i) for i in S])
    I = indlist.dot(2**np.arange(indlist.size)[::-1])
    return int(I)


def IndToState(I):
    strRep = np.binary_repr(I, width=3)
    ss = [i for i in strRep]
    S = [[int(i == '1'),int(i == '0')] for i in strRep]
    return S

