import numpy as np
import matplotlib.pyplot as plt
import binarystate as bst
import progress, models, rhs, rk4
import entanglement as ent

J = [2,1,3]

#spU = np.transpose([1,0])
#spD = np.transpose([0,1])

#initial conditions
a = np.zeros(8)
a[0] = 1/np.sqrt(4)
a[2] = 1/np.sqrt(4) 
a[7] = 1/np.sqrt(4)
a[4] = 1/np.sqrt(4)

paulix = np.array([(0,1),(1,0)]) 
pauliy = np.array([(0,-1j),(1j,0)])  
pauliz = np.array([(1,0),(0,-1)])
id2 = np.matrix(np.eye(2))

#spin-spin
A1 = np.array([paulix, paulix, pauliz])
A2 = np.array([pauliy, paulix, pauliz])
A3 = np.array([pauliz, pauliz, paulix])
A4 = np.array([id2, paulix, paulix])

#zeeman
Z1 = np.array([pauliz, id2, id2])
Z2 = np.array([id2, pauliz, id2])
Z3 = np.array([id2, id2, pauliz])

H = np.array([Z1, Z2, Z3, A1, A2, A3, A4])

T = 10

h_t = 0.1
m_t = int(T/h_t+1)
sol = np.zeros([m_t,8], dtype=np.complex128)
tretangel = np.zeros(m_t)
tretangel[0] = ent.entanglement(a)
sol[0] = a
t = 0
progress.start_progress("Calculating...")

for tid in range(m_t-1):
    x = tid/m_t*100
    progress.progress(x)
    a, t = rk4.step(rhs.rhs, a, t, h_t, H)
    sol[tid+1,:] = a
    tretangel[tid+1] = ent.entanglement(a)
    #print(a)

progress.end_progress()
ABS = np.array([np.absolute(i)**2 for i in sol[:]])

fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].set_ylim([0,1])
    axs[i].plot(range(m_t),ABS[:,i])

#plt.figure('Normalisering')
#superpos = np.array([np.sum(ABS[i,:]) for i in range(m_t)])
#plt.plot(range(m_t),superpos)

plt.figure('Tretangel')
plt.plot(range(m_t),tretangel)
plt.show()
