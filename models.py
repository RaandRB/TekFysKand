import numpy as np
paulix = np.array([(0,1),(1,0)]) 
pauliy = np.array([(0,-1j),(1j,0)])  
pauliz = np.array([(1,0),(0,-1)])
id2 = np.matrix(np.eye(2))

def heisenbergLine():
    #1 och 2
    A1 = np.array([paulix, paulix, id2])
    A2 = np.array([pauliy, pauliy, id2])
    A3 = np.array([pauliz, pauliz, id2])

    #2 och 3
    A4 = np.array([id2, paulix, paulix])
    A5 = np.array([id2, pauliy, pauliy])
    A6 = np.array([id2, pauliz, pauliz])

    return A1,A2,A3,A4,A5,A6

def heisenbergCircle():
    #1 och 2
    A1 = np.array([paulix, paulix, id2])
    A2 = np.array([pauliy, pauliy, id2])
    A3 = np.array([pauliz, pauliz, id2])

    #2 och 3
    A4 = np.array([id2, paulix, paulix])
    A5 = np.array([id2, pauliy, pauliy])
    A6 = np.array([id2, pauliz, pauliz])

    #1 och 3
    A7 = np.array([paulix, id2, paulix])
    A8 = np.array([pauliy, id2, pauliy])
    A9 = np.array([pauliz, id2, pauliz])

    return A1,A2,A3,A4,A5,A6,A7,A8,A9
