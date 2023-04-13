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

def threeParticle1(d=1):
    #s1*(s2 x s3)
    #x
    A1 = d*np.array([paulix,pauliy,pauliz])
    A2 = d*-1*np.array([paulix,pauliz,pauliy])
    #y
    A3 = d*-1*np.array([pauliy,paulix,pauliz])
    A4 = d*np.array([pauliy,pauliz,paulix])
    #z
    A5 = d*np.array([pauliz,paulix,pauliy])
    A6 = d*-1*np.array([pauliz,pauliy,paulix])

    return np.array([A1,A2,A3,A4,A5,A6])


def threeParticle2(d=1):
    #s2*(s1 x s3)
    #x
    A1 = d*np.array([pauliy,paulix,pauliz])
    A2 = d*-1*np.array([pauliz,paulix,pauliy])
    #y
    A3 = d*-1*np.array([paulix,pauliy,pauliz])
    A4 = d*np.array([pauliz,pauliy,paulix])
    #z
    A5 = d*np.array([paulix,pauliz,pauliy])
    A6 = d*-1*np.array([pauliy,pauliz,paulix])

    return np.array([A1,A2,A3,A4,A5,A6])

def threeParticle3(d=1):
    #s3*(s1 x s2)
    #x
    A1 = d*np.array([pauliy,pauliz,paulix])
    A2 = d*-1*np.array([pauliz,pauliy,paulix])
    #y
    A3 = d*-1*np.array([paulix,pauliz,pauliy])
    A4 = d*np.array([pauliz,paulix,pauliy])
    #z
    A5 = d*np.array([paulix,pauliy,pauliz])
    A6 = d*-1*np.array([pauliy,paulix,pauliz])

    return np.array([A1,A2,A3,A4,A5,A6])
