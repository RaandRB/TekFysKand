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
a = np.ones(8, dtype=np.complex128)
om = np.exp(2j/3*np.pi)
a[0] = -1
a[6] = -1
a[5] = 1
a = a/np.linalg.norm(a)

a0 = a

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
id2 = np.matrix(np.eye(2))

Zeeman = sh.Ham(pauliz,id2,id2) + sh.Ham(id2,pauliz,id2) + sh.Ham(id2,id2,pauliz)

H = models2.DMLine([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]) + 0.1*models2.heisenbergLine() + 0.01*Zeeman

w, v = np.linalg.eig((H))
w = np.real(w)
v = np.resize(v, (8,8))

w3 = np.abs(w)
w3 = np.sort(w3)
index = np.lexsort((w, np.abs(w)))
low_states = np.array([np.array(state) for ind,state in enumerate(v) if ind >= 4])
high_states = np.array([np.array(state) for ind,state in enumerate(v) if ind < 4 ])

print(w)

eigweights = np.array([np.abs(np.conjugate(a0)@A)**2 for A in v])

time_scales = np.array([2*np.pi/i for i in w])

coh = np.zeros((8,8))

for j in range(8):
    for i in range(8):
        coh[j][i] = np.round(np.abs(2*np.pi/(w[j]-w[i])),5)
coh = np.unique(np.resize(coh, 64))

temp1 = np.zeros((8,8))
for j in range(8):
    for i in range(8):
        temp1[j][i] = np.round(np.abs(w[j]+w[i]),5)
temp1 = np.unique(np.resize(temp1,64))

temp2 = np.zeros((np.size(temp1),np.size(temp1)))

for j in range(np.size(temp1)):
    for i in range(np.size(temp1)):
        temp2[j][i] = np.round(np.abs(2*np.pi/(temp1[j]-temp1[i])),4)

temp2 = np.unique(np.resize(temp2,np.size(temp1)))
print(temp2)

print(coh)
T_end = 1000
h_t = 0.01
m_t = int(T_end/h_t+1)
T = np.linspace(0, T_end, m_t)
sol = np.zeros([m_t,8], dtype=np.complex128)
tretangel = np.zeros(m_t)
tretangel[0] = ent.entanglement(np.flip(a0))
prob = np.zeros(m_t)
entropy = np.zeros(m_t)
entropy[0] = sh.entropy(a)
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
    entropy[tid+1] = sh.entropy(a)


progress.end_progress()


ABS = np.array([np.absolute(i) for i in sol[:]])
coh = np.sort(coh)
fourier2 = np.fft.fft(tretangel)
xfour2 = np.fft.fftfreq(m_t)

#plt.figure('Entropy')
#plt.plot(T, entropy)

plt.figure('eigenshurdas')
plt.plot(coh, '*')
print('Score: ' + str(np.size(coh)))

period1 = 100*int(coh[0]//(2*h_t))
period2 = 100*period1

test = np.zeros(m_t)
test[0:period1] = 0
test[-period1:] = 0
print(m_t-2*period1)
print(np.array(range(period1,m_t-period1)))
#for tid in range(period1,m_t-period1):
#    test[tid] = np.sqrt(2)/(period1)*np.trapz(prob[tid-period1:tid+period1])
#test2 = np.zeros(m_t-period1-2*period2)
#for tid in range(m_t-period1-period2,m_t+period2-period1):
#    test2[tid] = 1/(2*100*period2)*np.trapz(test[tid-period2:tid+period2])

#print((np.size(temp1),m_t))
#fourier1 = np.fft.fft(test[period1:-period1])
#xfour1 = np.fft.fftfreq(m_t-2*period1)

#plt.figure('Fourier')
#plt.plot(xfour1,fourier1)


plt.figure('Residual entanglement')
plt.plot(T, np.abs(tretangel),'r', label='Residual entanglement')
#for i in temp2:
#    if i < 4000:
#        plt.axvline(x=np.abs(i), color='r')
 

plt.figure('Survival probability')
plt.plot(T, prob, 'b')
#plt.plot(T,test, 'r')

colors = ['r', 'y', 'b', 'm', 'g', 'r', 'k', 'c']

#for i in w:
#    if 1/i < 4000:
#        plt.axvline(x=np.abs(2*np.pi/i), color='r')
    #plt.axhline(y=eigweights[i], color='b')
for i in coh:
    if i < 4000:
        plt.axvline(x=np.abs(i), color='g')
 

ax = plt.gca()
ax.legend()

plt.show()


