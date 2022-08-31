# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:58:51 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse.linalg import lsmr
from matplotlib import cm
import matplotlib.patches as mpatches
from math import log
from matplotlib.animation import FuncAnimation, PillowWriter

G = lambda u: 1/10*(u**4 - 2*u**2 + 0.25*u) #u**2 + u
#DG = lambda u: 2*u

sigNoise = 1
sigPrior = 4
y = -2#2

Phi = lambda u: 1/(2*sigNoise**2)*(y-G(u))**2 #+ 1/(2*sigPrior**2)*u**2
DPhi = lambda u: 1/sigNoise**2 * DG(u)*(G(u)-y) #+ 1/(sigPrior**2)*u


r = 0.5
kernel1 = lambda x1, x2: 1/(1+np.linalg.norm(x1-x2)**2/r**2)
kernel2 = lambda x1, x2: np.exp(-np.linalg.norm(x1-x2)**2/r**2)
kernel0 = lambda x1, x2: 1

kernelized = True

us =  np.random.normal(0,sigPrior,20)

def computeMoments_kernel(us, kernel):
    # compute local means
    J = len(us)
    ms = np.zeros_like(us,dtype=float)
    Gms = np.zeros_like(us, dtype=float)
    for i in range(len(us)):
        normalization = 0
        mean = 0
        mean_G = 0
        for j in range(len(us)):
            mean += kernel(us[i], us[j])*us[j]
            mean_G += kernel(us[i], us[j])*G(us[j])
            normalization += kernel(us[i], us[j])
        ms[i] = mean/normalization
        Gms[i] = mean_G/normalization
    Cs = np.zeros(J)
    Ds = np.zeros(J)
    Es = np.zeros(J)
    for i in range(J):
        normalization = 0
        C = 0
        D = 0
        E = 0
        for j in range(len(us)):
            C += kernel(us[i], us[j])*(us[j]-ms[i])**2
            D += kernel(us[i], us[j])*(us[j]-ms[i])*(G(us[j])-Gms[i])
            E += kernel(us[i], us[j])*(G(us[j])-Gms[i])**2
            normalization += kernel(us[i], us[j])
        Cs[i] = C/normalization
        Ds[i] = D/normalization
        Es[i] = E/normalization
    return ms, Cs, Ds, Es

def computeMoments(us):
    J = len(us)
    ms = np.mean(us)
    C_complete = np.cov(us, G(us))*(J-1)/J
    Cuu = C_complete[0,0]
    Cup = C_complete[0,1]
    Cpp = C_complete[1,1]
    return ms, Cuu, Cup, Cpp


def EKI_step(us, tau):
    ms, Cuu, Cup, Cpp = computeMoments(us)
    return us - tau*Cup*(tau*Cpp + sigNoise**(-2))**(-1)*(G(us)-y)

def EKI_kernel_step(us, kernel, tau):
    ms, Cs, Ds, Es = computeMoments_kernel(us, kernel)
    return us - tau*Ds*(tau*Es + sigNoise**(-2))**(-1)*(G(us)-y)

T = 500
N = 500
tau = T/N
us_vec = np.zeros((N,len(us)))
us_vec[0,:] = us

xplot = np.linspace(-4,4,100)

I = lambda x: Phi(x) #+ 0.5/sigPrior**2*(x)**2

yplot = np.exp(-I(xplot))
plt.figure()
plt.plot(xplot, yplot)


Z = np.trapz(yplot, xplot)
I = lambda x: Phi(x) #+ 0.5/sigPrior**2*(x)**2 + log(Z)
yplot = np.exp(-I(xplot))

fig, ax = plt.subplots(1,1)
p0 = ax.plot(xplot,yplot)
#p00 = ax.plot(xplot, np.exp(-Phi(xplot)), 'r-')
p1 = ax.plot(us_vec[0,:], np.exp(-I(us_vec[0,:])), '.')
ax.set_xlim([-4,4])
ax.set_ylim([0,max(yplot)])

def init():
    ax.set_xlim([-4,4])
    ax.set_ylim([0,max(yplot)])
    p0 = ax.plot(xplot,yplot)
    return p0, p1,

def update(frame_num): # expects frame to contain the new position of the ensemble
    xdata1 = us_vec[frame_num, :]
    ydata1 = np.exp(-I(us_vec[frame_num, :]))
    p1[0].set_data(xdata1, ydata1)
    ax.set_title(f"t={frame_num*T/N:.2f}")
    return p0, p1, 
for n in range(N-1):
    if kernelized:
        us_vec[n+1, :] = EKI_kernel_step(us_vec[n, :], kernel1, tau)
    else:
        us_vec[n+1, :] = EKI_step(us_vec[n, :], tau)
    #p1[0].set_data(us_vec[n+1,:], np.exp(-I(us_vec[n+1,:])))
    #plt.pause(0.1)
    #plt.show()


ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=20)

#%%
f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation_EKI_poly_long.gif" 
writergif = PillowWriter(fps=30) 
ani.save(f, writer=writergif)
    
    
    
    
    
    
    