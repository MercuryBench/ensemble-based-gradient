# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:40:42 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, pi
import scipy.linalg
from grad_inference import *
import time

np.random.seed(1)

adaptivetau = True

alpha = 1.0
J = 3
N_sim = 500

#-----------------------------------
# test case "banana"
# y = 0
# sigNoise = 0.5
# sigPrior = 2
# tau = 0.01
# u0 = np.random.normal(0,1.0,(2,J))

# xmin = -2
# xmax = 7
# ymin= -1
# ymax= 5


# G = lambda u: ((u[1]-2)**2-(u[0]-3.5) -1)
# Phi = lambda u: 0.5/(sigNoise**2)*(G(u)-y)**2


#-----------------------------------
#test case "Gaussian"
# y = 0
# sigNoise = 0.5
# sigPrior = 2
# tau = 0.03
# u0 = np.random.normal(0,1.0,(2,J))

# xmin = -3
# xmax = 3
# ymin= -3
# ymax= 3


# G = lambda u: u[0] + u[1]
# Phi = lambda u: 0.5/(sigNoise**2)*(G(u)-y)**2

#-----------------------------------
# test case "ellipt2d"
y = np.array([27.5,79.7])
sigNoise = 0.4
sigPrior = 10
tau = 0.01
#
u0 = np.random.normal(0,0.5,(2,J)) + np.array([[-3.0],[102]])

xmin = -3.75
xmax = -1.75
ymin =  100
ymax =  106

xobs = np.array([0.25, 0.75])



p = lambda x, u: np.tensordot(u[1],x,axes=0) + np.tensordot(np.exp(-u[0]),(-x**2/2+x/2), axes=0)
Dp = lambda x, u: np.stack((-np.tensordot((-x**2/2+x/2),np.exp(-u[0]), axes=0),x[:,np.newaxis]+0*u[1]))
G = lambda u: p(xobs, u)
DG = lambda u: Dp(xobs, u)
Phi = lambda u: 0.5/(sigNoise**2)*np.linalg.norm(G(u)-y, axis=-1)**2
DPhi = lambda u: 1/sigNoise**2*np.einsum('ijk,kj->ik',DG(u),G(u)-y)


# parameters independent of test case



I = lambda u: Phi(u) + 0.5/(sigPrior**2)*np.linalg.norm(u,axis=0)**2
DI = lambda u: DPhi(u) + 1/sigPrior**2*u

#%%


# method for computing all gradients at once
def compute_gradients(points):
    vs = np.zeros_like(points)
    d, J = points.shape
    H = np.zeros((points.shape[0],points.shape[0],points.shape[1]))
    for i in range(J):
        return_dict= inferGradientAndHess(points, I(points), hessian = True, ind=i, additionalvariance=0.0)
        vs[:,i], H[:,:,i] = return_dict['grad'], return_dict['H']
    return vs, H

def compute_gradients_withmemory(points, memory):
    vs = np.zeros_like(points)
    d, J = points.shape
    H = np.zeros((points.shape[0],points.shape[0],points.shape[1]))
    all_points = np.concatenate((points,memory), axis=1)
    all_vals = I(all_points)
    for i in range(J):
        return_dict= inferGradientAndHess(all_points, all_vals, hessian = True, ind=i, additionalvariance=0.0)
        vs[:,i], H[:,:,i] = return_dict['grad'], return_dict['H']
    return vs, H
    

N_sim = 10

us_list = np.zeros((2,J,N_sim))
us_list[:,:,0] = u0
memory = None
for n in range(N_sim-1):
    us = us_list[:,:,n]    
    j = n % J
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    ensembleplusmean = np.concatenate((m_us,us),axis=1)ensembleplusmean = np.concatenate((m_us,us,memory),axis=1)
    return_dict= inferGradientAndHess(ensembleplusmean, I(ensembleplusmean), hessian = True, ind=0, additionalvariance=1.0)
    vs, H = return_dict['grad'], return_dict['H']
    new_pos = m_us.flatten() - np.linalg.solve(H,vs)
    us_list[:,:,n+1] = us
    us_list[:,j,n+1] = new_pos
    
u0s = np.linspace(xmin,xmax,150)
u1s = np.linspace(ymin,ymax,150)
U0, U1 = np.meshgrid(u0s,u1s)
U = np.stack((U0,U1))


plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
plt.contourf(U0, U1, np.exp(-I(U)), 10)

#%%
ind = 6
plt.scatter(us_list[0,:,ind], us_list[1,:,ind])
    
    
    
    
    
    


