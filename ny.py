import numpy as np
import matplotlib.pyplot as plt
import binarystate as bst
import progress, models2, rk4
import rhs2, rhs
import entanglement as ent
import shurda as sh
import fractions

#initial conditions
a = np.zeros(8, dtype=np.complex128)
a[0] = 1/np.sqrt(2)
a[7] = 1/np.sqrt(2)


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

Zeeman = sh.Ham(pauliz,id2,id2) + sh.Ham(id2,pauliz,id2) + sh.Ham(id2,id2,pauliz)


H = models2.DMCircle([1,1,0])
w, v = np.linalg.eig(H)
w = np.real(w)

#print(w)
#ratio1 = w[0]/w[7]
#ratio2 = w[0]/w[3]
#test1 = fractions.Fraction(ratio1)
#test2 = fractions.Fraction(ratio2)
#print((test1, test2))

#numerator1 = test1.numerator
#numerator2 = test2.numerator
#prime1 = sh.prime_factors(np.abs(numerator1))
#prime2 = sh.prime_factors(np.abs(numerator2))

#print(prime1)
#print(prime2)


#A = models.DM([1,0,0])
T_end = 10
h_t = 0.01
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
    #if prob[tid+1] >= 0.98:
    #    print(tid)

progress.end_progress()

ABS = np.array([np.absolute(i) for i in sol[:]])

fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].set_ylim([0,1])
    axs[i].plot(range(m_t),ABS[:,i])

fig2, axs2 = plt.subplots(8)
for i in range(8):
    axs2[i].set_ylim([-1,1])
    axs2[i].plot(range(m_t), np.real(sol[:,i]), 'r')
    axs2[i].plot(range(m_t), np.imag(sol[:,i]), 'b')

diff = np.absolute(prob-tretangel)

plt.figure('Tretangel och Prob')
plt.plot(range(m_t), tretangel,'r', label='Residual entanglement')
plt.plot(range(m_t), prob, 'b', label='Survival probability')
plt.legend(loc='upper left')

plt.figure('Diff')
plt.plot(range(m_t), diff)

plt.show()

