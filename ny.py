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
a[0:8] = 1

a = a/np.linalg.norm(a)

a0 = a

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
id2 = np.matrix(np.eye(2))

Zeeman = sh.Ham(pauliz,id2,id2) + sh.Ham(id2,pauliz,id2) + sh.Ham(id2,id2,pauliz)

H = models2.DMLine() + 0.01*Zeeman + 0.01*models2.heisenbergLine()

w, v = np.linalg.eig((H))
w = np.real(w)
#for i in v:
#    print(np.transpose(a)@np.resize(i,8))
print(w)


#fractions = [fr.Fraction(Decimal(str(i))) for i in w2]

#m = [i.denominator for i in fractions]
#n = [i.numerator for i in fractions]


#print(fractions)
#lcm = 1
#for i in n:
#    lcm = lcm*i//np.gcd(lcm, i)
#gcd = math.gcd(*m)
#print((lcm,gcd))
#revivalTime = 2*np.pi*lcm/gcd
#print(revivalTime)
w3 = np.abs(w)
w3 = np.sort(w3)

print(w3)

count1 = sum(i<0 for i in w3[0:4])
count2 = sum(i<0 for i in w3[4:8])


lowerw = np.array(w3[0:4])
upperw = np.array(w3[4:8])
plw = np.polyfit(np.array(range(4)), lowerw, 3)
puw = np.polyfit(np.array(range(4)), upperw, 3)


test_t1 = 2*np.pi/lowerw[0]
test_t2 = 2*np.pi/upperw[0]

kep_t1 = 2*np.pi/lowerw[1]
kep_t2 = 2*np.pi/upperw[1]

res_t1 = 2*np.pi/(1/2*lowerw[2])
res_t2 = 2*np.pi/(1/2*upperw[2])

supres_t1 = 2*np.pi/(1/6*lowerw[3])
supres_t2 = 2*np.pi/(1/6*upperw[3])

print(test_t1, test_t2)

print(kep_t1, kep_t2)

print(res_t1, res_t2)

print(supres_t1, supres_t2)

T_end = 2000
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
progress.start_progress("Calculating")

for tid in range(m_t-1):
    x = tid/m_t*100
    progress.progress(x)
    a, t = rk4.step(rhs2.rhs, a, t, h_t, H)
    sol[tid+1,:] = np.flip(a)
    tretangel[tid+1] = ent.entanglement(np.flip(a))
    prob[tid+1] = np.abs((np.conjugate(a0)@a))**2
    #if prob[tid+1] >= 0.98:
    #    print(tid)

progress.end_progress()

ABS = np.array([np.absolute(i) for i in sol[:]])

#fig, axs = plt.subplots(8)
#for i in range(8):
#    axs[i].set_ylim([0,1])
#    axs[i].plot(range(m_t),ABS[:,i])

#fig2, axs2 = plt.subplots(8)
#for i in range(8):
#    axs2[i].set_ylim([-1,1])
#    axs2[i].plot(range(m_t), np.real(sol[:,i]), 'r')
#    axs2[i].plot(range(m_t), np.imag(sol[:,i]), 'b')


#sq = np.array([np.sum(ABS[i,:]**2) for i in range(m_t)])
#plt.figure('Normalisering')
#ax = plt.gca()
#ax.set_ylim([0,1.1])
#plt.plot(range(m_t), sq)

diff = np.absolute(prob-tretangel)

plt.figure('Residual entanglement')
plt.plot(T, tretangel,'r', label='Residual entanglement')
plt.figure('Survival probability')
plt.plot(T, prob, 'b', label='Survival probability')
plt.axvline(x=test_t1, color='y')
plt.axvline(x=test_t2, color='r')
plt.axvline(x=kep_t1, color ='y')
plt.axvline(x=kep_t2, color='r')
plt.axvline(x=res_t1, color='y')
plt.axvline(x=res_t2, color='r')
plt.axvline(x=supres_t1, color='y')
plt.axvline(x=supres_t2, color='r')
#plt.legend(loc='upper left')



#plt.figure('Diff')
#plt.plot(range(m_t), diff)

plt.show()

