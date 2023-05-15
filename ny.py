import numpy as np
import matplotlib.pyplot as plt
import binarystate as bst
import progress, models2, rk4
import rhs2, rhs
import entanglement as ent
import shurda as sh
import fractions as fr
import math
from decimal import Decimal

#initial conditions
a = np.zeros(8, dtype=np.complex128)
om = np.exp(2j/3*np.pi)
a[0] = 1
a[7] = 1
a = a/np.linalg.norm(a)

a0 = a

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
id2 = np.matrix(np.eye(2))

Zeeman = sh.Ham(pauliz,id2,id2) + sh.Ham(id2,pauliz,id2) + sh.Ham(id2,id2,pauliz)

H = models2.DMCircle() + 0.01*models2.heisenbergCircle() + 0.01*Zeeman

w, v = np.linalg.eig(H)
w = np.real(w)
v = np.resize(v, (8,8))

print(w)

coh = np.zeros((8,8))

for j in range(8):
    for i in range(8):
        coh[j][i] = np.round(np.abs(2*np.pi/(w[j]-w[i])),5)
coh = np.unique(np.resize(coh, 64))

print(coh)
T_end = 1000
h_t = 0.01
m_t = int(T_end/h_t+1)
T = np.linspace(0, T_end, m_t)
sol = np.zeros([m_t,8], dtype=np.complex128)
tretangel = np.zeros(m_t)
tretangel[0] = ent.entanglement(np.flip(a0))
prob = np.zeros(m_t)
prob[0] = 1
sol[0] = np.flip(a)
t = 0
progress.start_progress("Calculating")

for tid in range(m_t-1):
    x = tid/m_t*100
    progress.progress(x)
    a, t = rk4.step(rhs2.rhs, a, t, h_t, H)
    sol[tid+1,:] = np.flip(a)
    tretangel[tid+1] = ent.entanglement(np.flip(a))
    prob[tid+1] = np.abs((np.conjugate(a0)@a))**2


progress.end_progress()

ABS = np.array([np.absolute(i) for i in sol[:]])
coh = np.sort(coh)


plt.figure('eigenshurdas')
plt.plot(coh, '*')


plt.figure('Residual entanglement')
plt.plot(T, np.abs(tretangel),'r', label='Residual entanglement')
 

plt.figure('Survival probability')
plt.plot(T, prob, 'b')

colors = ['r', 'y', 'b', 'm', 'g', 'r', 'k', 'c']

for i in coh:
    if i < 4000:
        plt.axvline(x=np.abs(i), color='g')
 

ax = plt.gca()
ax.legend()

plt.show()


