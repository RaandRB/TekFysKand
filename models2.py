import numpy as np
import shurda as sh

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
id2 = np.matrix(np.eye(2))

def DMCircle(d=[0,0,1]):
    #s1 x s2
    #x
    A1_1 = d[0]*sh.Ham(pauliy,pauliz,id2)
    A1_2 = d[0]*-1*sh.Ham(pauliz,pauliy,id2)
    #y
    A1_3 = d[1]*-1*sh.Ham(paulix,pauliz,id2)
    A1_4 = d[1]*sh.Ham(pauliz,paulix,id2)
    #z
    A1_5 = d[2]*sh.Ham(paulix,pauliy,id2)
    A1_6 = d[2]*-1*sh.Ham(pauliy,paulix,id2)

    #s1 x s3
    #x
    A2_1 = d[0]*sh.Ham(pauliy,id2,pauliz)
    A2_2 = d[0]*-1*sh.Ham(pauliz,id2,pauliy)
    #y
    A2_3 = d[1]*-1*sh.Ham(paulix,id2,pauliz)
    A2_4 = d[1]*sh.Ham(pauliz,id2,paulix)
    #z
    A2_5 = d[2]*sh.Ham(paulix,id2,pauliy)
    A2_6 = d[2]*-1*sh.Ham(pauliy,id2,paulix)

    #s2 x s3
    #x
    A3_1 = d[0]*sh.Ham(id2,pauliy,pauliz)
    A3_2 = d[0]*-1*sh.Ham(id2,pauliz,pauliy)
    #y
    A3_3 = d[1]*-1*sh.Ham(id2,paulix,pauliz)
    A3_4 = d[1]*sh.Ham(id2,pauliz,paulix)
    #z
    A3_5 = d[2]*sh.Ham(id2,paulix,pauliy)
    A3_6 = d[2]*-1*sh.Ham(id2,pauliy,paulix)

    return A1_1+A1_2+A1_3+A1_4+A1_5+A1_6+A2_1+A2_2+A2_3+A2_4+A2_5+A2_6+A3_1+A3_2+A3_3+A3_4+A3_5+A3_6





