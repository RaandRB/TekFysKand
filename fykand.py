import numpy as np
import matplotlib.pyplot as plt
import binarystate as bst
import progress, models, rhs, rk4
import entanglement as ent


#initial conditions
a = np.zeros(8)
a[0] = 1/np.sqrt(4) 
a[7] = 1/np.sqrt(4)
a[1] = 1/np.sqrt(4)
a[5] = 1/np.sqrt(4)
paulix = np.array([(0,1),(1,0)]) 
pauliy = np.array([(0,-1j),(1j,0)])  
pauliz = np.array([(1,0),(0,-1)])
id2 = np.matrix(np.eye(2))

#zeeman
Z1 = np.array([pauliz, id2, id2])
Z2 = np.array([id2, pauliz, id2])
Z3 = np.array([id2, id2, pauliz])

H = np.array([Z1, Z2, Z3])
#A = models.DM([1,0,0])
A = models.DMCircle([1,0,0])
A2 = models.threeParticleLine(
        A2 = models.threeParticleLine())
H = np.append(H, A, axis=0)

T_end = 10
h_t = 0.01
m_t = int(T_end/h_t+1)
T = np.linspace(0, T_end, m_t)
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

figfourier, axsfourier = plt.subplots(8)
for i in range(8):
    fourier = np.absolute(np.fft.fft(ABS[:,i]))
    freq = np.fft.fftfreq(m_t, h_t)
    axsfourier[i].set_xlim([-20,20])
    axsfourier[i].plot(freq, fourier)

#plt.figure('FFT')
#print(np.shape(ABS[:,0]))
#fourier = np.fft.fft(ABS[:,0])
#freq = np.fft.fftfreq(m_t)
#print(freq)
#print(fourier)
#plt.plot(freq, np.absolute(fourier))


fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].set_ylim([0,1])
    axs[i].plot(T,ABS[:,i])

plt.figure('Normalisering')
superpos = np.array([np.sum(ABS[i,:]) for i in range(m_t)])
plt.plot(range(m_t),superpos)

plt.figure('Tretangel')
plt.plot(range(m_t),tretangel)

plt.figure('Tretangel fourier')
four = np.fft.fft(tretangel)
ffreq = np.fft.fftfreq(m_t, h_t)
plt.plot(ffreq, np.absolute(four))
plt.ylim([0,50])
plt.show()


