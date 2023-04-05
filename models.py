import numpy as np
paulix = np.array([(0,1),(1,0)]) 
pauliy = np.array([(0,-1j),(1j,0)])  
pauliz = np.array([(1,0),(0,-1)])
id2 = np.matrix(np.eye(2))

def heisenbergLine(J=[1,1,1]):
    #1 och 2
    A1_1 = J[0]*np.array([paulix, paulix, id2])
    A1_2 = J[0]*np.array([pauliy, pauliy, id2])
    A1_3 = J[0]*np.array([pauliz, pauliz, id2])

    #2 och 3
    A2_1 = J[1]*np.array([id2, paulix, paulix])
    A2_2 = J[1]*np.array([id2, pauliy, pauliy])
    A2_3 = J[1]*np.array([id2, pauliz, pauliz])

    return np.array([A1_1,A1_2,A1_3,A2_1,A2_2,A2_3])

def heisenbergCircle(J=[1,1,1]):
    #1 och 2
    A1_1 = J[0]*np.array([paulix, paulix, id2])
    A1_2 = J[0]*np.array([pauliy, pauliy, id2])
    A1_3 = J[0]*np.array([pauliz, pauliz, id2])

    #2 och 3
    A2_1 = J[1]*np.array([id2, paulix, paulix])
    A2_2 = J[1]*np.array([id2, pauliy, pauliy])
    A2_3 = J[1]*np.array([id2, pauliz, pauliz])

    #1 och 3
    A3_1 = J[2]*np.array([paulix, id2, paulix])
    A3_2 = J[2]*np.array([pauliy, id2, pauliy])
    A3_3 = J[2]*np.array([pauliz, id2, pauliz])

    return np.array([A1_1,A1_2,A1_3,A2_1,A2_2,A2_3,A3_1,A3_2,A3_3])

def DMLine(d=[0,0,1]):
    #s1 x s2
    #x
    A1_1 = d[0]*np.array([pauliy,pauliz,id2])
    A1_2 = d[0]*-1*np.array([pauliz,pauliy,id2])
    #y
    A1_3 = d[1]*-1*np.array([paulix,pauliz,id2])
    A1_4 = d[1]*np.array([pauliz,paulix,id2])
    #z
    A1_5 = d[2]*np.array([paulix,pauliy,id2])
    A1_6 = d[2]*-1*np.array([pauliy,paulix,id2])

    #s1 x s3
    #x
    A2_1 = d[0]*np.array([pauliy,id2,pauliz])
    A2_2 = d[0]*-1*np.array([pauliz,id2,pauliy])
    #y
    A2_3 = d[1]*-1*np.array([paulix,id2,pauliz])
    A2_4 = d[1]*np.array([pauliz,id2,paulix])
    #z
    A2_5 = d[2]*np.array([paulix,id2,pauliy])
    A2_6 = d[2]*-1*np.array([pauliy,id2,paulix])
    
    return np.array([A1_1,A1_2,A1_3,A1_4,A1_5,A1_6,A2_1,A2_2,A2_3,A2_4,A2_5,A2_6])

def DMCircle(d=[0,0,1]):
    #s1 x s2
    #x
    A1_1 = d[0]*np.array([pauliy,pauliz,id2])
    A1_2 = d[0]*-1*np.array([pauliz,pauliy,id2])
    #y
    A1_3 = d[1]*-1*np.array([paulix,pauliz,id2])
    A1_4 = d[1]*np.array([pauliz,paulix,id2])
    #z
    A1_5 = d[2]*np.array([paulix,pauliy,id2])
    A1_6 = d[2]*-1*np.array([pauliy,paulix,id2])

    #s1 x s3
    #x
    A2_1 = d[0]*np.array([pauliy,id2,pauliz])
    A2_2 = d[0]*-1*np.array([pauliz,id2,pauliy])
    #y
    A2_3 = d[1]*-1*np.array([paulix,id2,pauliz])
    A2_4 = d[1]*np.array([pauliz,id2,paulix])
    #z
    A2_5 = d[2]*np.array([paulix,id2,pauliy])
    A2_6 = d[2]*-1*np.array([pauliy,id2,paulix])

    #s2 x s3
    #x
    A3_1 = d[0]*np.array([id2,pauliy,pauliz])
    A3_2 = d[0]*-1*np.array([id2,pauliz,pauliy])
    #y
    A3_3 = d[1]*-1*np.array([id2,paulix,pauliz])
    A3_4 = d[1]*np.array([id2,pauliz,paulix])
    #z
    A3_5 = d[2]*np.array([id2,paulix,pauliy])
    A3_6 = d[2]*-1*np.array([id2,pauliy,paulix])
    
    return np.array([A1_1,A1_2,A1_3,A1_4,A1_5,A1_6,A2_1,A2_2,A2_3,A2_4,A2_5,A2_6,A3_1,A3_2,A3_3,A3_4,A3_5,A3_6])

def threeParticleLine(d=[1,1,1]):
    #s3*(s1 x s2)
    #x
    A1_1 = d[0]*np.array([pauliy,pauliz,paulix])
    A1_2 = d[0]*-1*np.array([pauliz,pauliy,paulix])
    #y
    A1_3 = d[0]*-1*np.array([paulix,pauliz,pauliy])
    A1_4 = d[0]*np.array([pauliz,paulix,pauliy])
    #z
    A1_5 = d[0]*np.array([paulix,pauliy,pauliz])
    A1_6 = d[0]*-1*np.array([pauliy,paulix,pauliz])

    #s2*(s1 x s3)
    #x
    A2_1 = d[0]*np.array([pauliy,paulix,pauliz])
    A2_2 = d[0]*-1*np.array([pauliz,paulix,pauliy])
    #y
    A2_3 = d[1]*-1*np.array([paulix,pauliy,pauliz])
    A2_4 = d[1]*np.array([pauliz,pauliy,paulix])
    #z
    A2_5 = d[2]*np.array([paulix,pauliz,pauliy])
    A2_6 = d[2]*-1*np.array([pauliy,pauliz,paulix])

    return np.array([A1_1,A1_2,A1_3,A1_4,A1_5,A1_6,A2_1,A2_2,A2_3,A2_4,A2_5,A2_6])


def threeParticleCircle(d=[1,1,1]):
    #s3*(s1 x s2)
    #x
    A1_1 = d[0]*np.array([pauliy,pauliz,paulix])
    A1_2 = d[0]*-1*np.array([pauliz,pauliy,paulix])
    #y
    A1_3 = d[0]*-1*np.array([paulix,pauliz,pauliy])
    A1_4 = d[0]*np.array([pauliz,paulix,pauliy])
    #z
    A1_5 = d[0]*np.array([paulix,pauliy,pauliz])
    A1_6 = d[0]*-1*np.array([pauliy,paulix,pauliz])

    #s2*(s1 x s3)
    #x
    A2_1 = d[0]*np.array([pauliy,paulix,pauliz])
    A2_2 = d[0]*-1*np.array([pauliz,paulix,pauliy])
    #y
    A2_3 = d[1]*-1*np.array([paulix,pauliy,pauliz])
    A2_4 = d[1]*np.array([pauliz,pauliy,paulix])
    #z
    A2_5 = d[2]*np.array([paulix,pauliz,pauliy])
    A2_6 = d[2]*-1*np.array([pauliy,pauliz,paulix])

    #s1*(s2 x s3)
    #x
    A3_1 = d[0]*np.array([paulix,pauliy,pauliz])
    A3_2 = d[0]*-1*np.array([paulix,pauliz,pauliy])
    #y
    A3_3 = d[1]*-1*np.array([pauliy,paulix,pauliz])
    A3_4 = d[1]*np.array([pauliy,pauliz,paulix])
    #z
    A3_5 = d[2]*np.array([pauliz,paulix,pauliy])
    A3_6 = d[2]*-1*np.array([pauliz,pauliy,paulix])

    return np.array([A1_1,A1_2,A1_3,A1_4,A1_5,A1_6,A2_1,A2_2,A2_3,A2_4,A2_5,A2_6,A3_1,A3_2,A3_3,A3_4,A3_5,A3_6])
