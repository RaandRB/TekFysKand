import numpy as np
import matplotlib.pyplot as plt
import rk4
import binarystate as bst
import rhs
import progress

#psi = (np.array([np.transpose([0,1]),np.transpose([0,1]),np.transpose([1,0])]))

spU = np.transpose([1,0])
spD = np.transpose([0,1])
a = np.zeros(8)
a[0] = 1
print(a)
psi = np.array([spD, spU, spD])
paulix = np.array([(0,1),(1,0)]) 
pauliy = np.array([(0,-1j),(1j,0)])  
pauliz = np.array([(1,0),(0,-1)])
id2 = np.matrix(np.eye(2))

A1 = np.array([pauliy, pauliy, pauliy])
A2 = np.array([id2, paulix, paulix])
A6 = np.array([pauliz, pauliz, pauliz])
A3 = np.array([pauliz, id2, id2])
A4 = np.array([id2, pauliz, id2])
A5 = np.array([id2, id2, pauliz])

A = A2@A1
#def rhs(U):
#    return -1j*(paulix + pauliy + pauliz)@U
H = np.array([A1,A2,A3,A4,A5,A6])
T = 10
h_t = 0.01
m_t = int(T/h_t+1)
sol = np.zeros([m_t,8], dtype=np.complex128)
sol[0] = a
t = 0
progress.start_progress("Calculating...")
for tid in range(m_t-1):
    x = tid/m_t*100
    progress.progress(x)
    a, t = rk4.step(rhs.rhs, a, t, h_t, H)
    sol[tid+1,:] = a
    #print(a)
progress.end_progress()
ABS = np.array([np.absolute(i)**2 for i in sol[:]])
fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].set_ylim([0,1])
    axs[i].plot(range(m_t),ABS[:,i])
plt.show()



