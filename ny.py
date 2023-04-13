import numpy as np
import matplotlib.pyplot as plt
import binarystate as bst
import progress, models2, rk4
import rhs2, rhs
import entanglement as ent
import shurda as sh

#initial conditions
a = np.zeros(8, dtype=np.complex128)
a[1] = 7
a[7] = 1
a[0] = -5
a = a/np.linalg.norm(a)

a0 = a

#psi = sh.initial(a)
#print(psi)

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
id2 = np.matrix(np.eye(2))

#zeeman
Z1 = np.array([pauliz, id2, id2])
Z2 = np.array([id2, pauliz, id2])
Z3 = np.array([id2, id2, pauliz])
ZZ = np.array([Z1,Z2,Z3])
XX = np.array([np.array([paulix, paulix, paulix])])

Zeeman = 8.8889*sh.Ham(pauliz,id2,id2) + 4.5*sh.Ham(id2,pauliz,id2) + sh.Ham(id2,id2,pauliz)


H = Zeeman + models2.DMCircle([np.pi,2,np.e])

w, v = np.linalg.eig(H)
w = np.real(w)

print(w)

print(w[0]/w[7])
print(w[0]/w[4])
print(w[0]/w[3])
print(w[4]/w[7])
print(w[3]/w[4])

#A = models.DM([1,0,0])
T_end = 100
h_t = 0.001
m_t = int(T_end/h_t+1)
T = np.linspace(0, T_end, m_t)
sol = np.zeros([m_t,8], dtype=np.complex128)
tretangel = np.zeros(m_t)
tretangel[0] = ent.entanglement(np.flip(a))
prob = np.zeros(m_t)
prob[0] = 1
sol[0] = np.flip(a)
t = 0
progress.start_progress("Calculating...")

for tid in range(m_t-1):
    x = tid/m_t*100
    progress.progress(x)
    a, t = rk4.step(rhs2.rhs, a, t, h_t, H)
    sol[tid+1,:] = np.flip(a)
    tretangel[tid+1] = ent.entanglement(np.flip(a))
    prob[tid+1] = np.abs((np.transpose(a0)@a))**2

progress.end_progress()

ABS = np.array([np.absolute(i) for i in sol[:]])

fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].set_ylim([0,1])
    axs[i].plot(range(m_t),ABS[:,i])

plt.figure('Tretangel')
plt.plot(range(m_t), tretangel)

plt.figure('Probability')
plt.plot(range(m_t), prob)


plt.show()

