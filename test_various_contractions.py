# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 19:35:27 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse.linalg import lsmr
from math import sqrt, pi, e, sqrt, ceil, exp
import time

from grad_inference import *
from matplotlib.animation import FuncAnimation, PillowWriter

### du = -lam*u + sig dW

np.random.seed(3)

lam = 1
sig = 2


N_it = 100
tau = 0.01
N_MC = 1000
ts = np.arange(N_it)*tau

d = 2

u0 = np.random.normal(0,1,d)

# vanilla
us_1 = np.zeros((d,N_it,N_MC))
us_1[:,0,:] = u0[:,np.newaxis]
for n in range(N_it-1):
    us_1[:,n+1,:] = us_1[:,n,:] - lam*tau*us_1[:,n,:] + sig*sqrt(tau)*np.linalg.norm(us_1[:,n,:], axis=0, keepdims=True)*np.random.normal(0,1,(d,N_MC))


if d==2:
    plt.figure()
    plt.plot(us_1[0,:,:], us_1[1,:,:])
    plt.plot(0,0,'ks')
    plt.title("vanilla 2d")
    plt.axis("equal")
    

plt.figure()
plt.semilogy(ts, 0.5*np.linalg.norm(us_1, axis=0)**2)
plt.semilogy(ts, 0.5*np.mean(np.linalg.norm(us_1, axis=0)**2, axis=-1), 'k', linewidth=3)
plt.title("vanilla")


logmeanrate = np.log(0.5*np.mean(np.linalg.norm(us_1, axis=0)**2, axis=-1))
lmr = (logmeanrate[-1]-logmeanrate[0])/(ts[-1]-ts[0])
plt.plot(ts,np.linalg.norm(u0)*  np.exp(lmr*ts))
print(f"lograte projected: {lmr} (theoretical: {-2*(lam-d*sig**2/(2))})")

# component
us_2 = np.zeros((d,N_it,N_MC))
us_2[:,0,:] = u0[:,np.newaxis]
for n in range(N_it-1):
    us_2[:,n+1,:] = us_2[:,n,:] - lam*tau*us_2[:,n,:] + sig*sqrt(tau)*us_2[:,n,:]*np.random.normal(0,1,(d,N_MC))




if d==2:
    plt.figure()
    plt.plot(us_2[0,:,:], us_2[1,:,:])
    plt.plot(0,0,'ks')
    plt.title("comp 2d")
    plt.axis("equal")

plt.figure()
plt.semilogy(ts, 0.5*np.linalg.norm(us_2, axis=0)**2)
plt.semilogy(ts, 0.5*np.mean(np.linalg.norm(us_2, axis=0)**2, axis=-1), 'k', linewidth=3)
plt.title("component")


logmeanrate = np.log(np.mean(np.linalg.norm(us_2, axis=0)**2, axis=-1))
lmr = (logmeanrate[-1]-logmeanrate[0])/(ts[-1]-ts[0])
plt.plot(ts,np.linalg.norm(u0)*  np.exp(lmr*ts))
print(f"lograte projected: {lmr} (theoretical: {-2*(lam-sig**2/(2))})")

# orthogonal
us_3 = np.zeros((d,N_it,N_MC))
us_3[:,0,:] = u0[:,np.newaxis]
for n in range(N_it-1):
    us = us_3[:,n,:]
    diff0 = sig*sqrt(tau)*np.linalg.norm(us, axis=0, keepdims=True)*np.random.normal(0,1,(d,N_MC))
    us_step = us + diff0/2
    
    diff_orth = diff0 - (np.einsum("ij,ij->j", us_step, diff0))/(np.einsum("ij,ij->j", us_step, us_step))*us_step
    
    us_3[:,n+1,:] = (us )*exp(-tau*lam)+ diff_orth
    
    
    # temp = us_3[:,n,:] - lam*tau*us_3[:,n,:] 
    # diff0 = sig*sqrt(tau)*np.linalg.norm(us_3[:,n,:], axis=0, keepdims=True)**np.random.normal(0,1,(d,N_MC))
    # us = us_3[:,n,:]
    # diff_orth = diff0 - (np.einsum("ij,ij->j", us, diff0))/(np.einsum("ij,ij->j", us, us))*us
    
    # us_3[:,n+1,:] = temp + diff_orth

# plt.figure()
# plt.plot(us_3[0,:,:])
# plt.plot(np.mean(np.linalg.norm(us_3, axis=0), axis=-1), 'k', linewidth=3)
# plt.title("projected")

# plt.figure()
# plt.plot(us_3[1,:,:])
# plt.plot(np.mean(np.linalg.norm(us_3, axis=0), axis=-1), 'k', linewidth=3)
# plt.title("projected")

if d==2:
    plt.figure()
    plt.plot(us_3[0,:,:], us_3[1,:,:])
    plt.plot(0,0,'ks')
    plt.title("proj 2d")
    plt.axis("equal")

plt.figure()
plt.semilogy(ts, 0.5*np.linalg.norm(us_3, axis=0)**2)
plt.semilogy(ts, 0.5*np.mean(np.linalg.norm(us_3, axis=0)**2, axis=-1), 'k', linewidth=3)
#plt.semilogy(ts, np.linalg.norm(u0)* np.exp(-lam*ts))
plt.plot()
plt.title("projected")

logmeanrate = np.log(np.mean(np.linalg.norm(us_3, axis=0)**2, axis=-1))
lmr = (logmeanrate[-1]-logmeanrate[0])/(ts[-1]-ts[0])
plt.plot(ts, np.linalg.norm(u0)* np.exp(lmr*ts))
print(f"lograte projected: {lmr} (theoretical: {-2*lam})")

    
    
if d==2:
    plt.figure()
    m1 = np.mean(us_1, axis=-1)
    m2 = np.mean(us_2, axis=-1)
    m3 = np.mean(us_3, axis=-1)
    plt.plot(m1[0], m1[1], label="vanilla")
    plt.plot(m2[0], m2[1], label="comp")
    plt.plot(m3[0], m3[1], label="orth")
    plt.plot(0,0,'rx')
    plt.legend()
    
    
    
    