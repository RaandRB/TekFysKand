import numpy as np
import rhs2 as rhs

spU = np.array([1,0])
spD = np.array([0,1])

def state3(q1,q2,q3):
    return np.kron(np.kron(q1,q2),q3)

s0 = state3(spD,spD,spD)
s1 = state3(spD,spD,spU)
s2 = state3(spD,spU,spD)
s3 = state3(spD,spU,spU)
s4 = state3(spU,spD,spD)
s5 = state3(spU,spD,spU)
s6 = state3(spU,spU,spD)
s7 = state3(spU,spU,spU)

states = np.array([s0,s1,s2,s3,s4,s5,s6,s7])

def initial(a):
    return [states[i]*a[i] for i in range(8)]

def entropy(A):
    ent = sum(np.abs(a)**2*np.log(np.abs(a)**2) for a in A)

    return -ent


def Ham(H1, H2, H3):
    return np.kron(np.kron(H1,H2),H3)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
