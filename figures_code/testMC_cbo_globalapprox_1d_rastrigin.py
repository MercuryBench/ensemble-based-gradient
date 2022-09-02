#
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:31:16 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse.linalg import lsmr
from math import sqrt, pi, e, exp
import time

import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)

from grad_inference import *
from cbo import *
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(2)


# Rastrigin
fnc = lambda x: 10 + x**2 - 10*np.cos(2*pi*x)
#Dfnc = lambda x, y: np.array([2*x + 2*pi*10*np.sin(2*pi*x),  2*y + 2*pi*10*np.sin(2*pi*y)])
#D2fnc = lambda x,y:np.array([[2 + 4*pi**2*10*np.cos(2*pi*x), 0], [0, 2 + 4*pi**2*10*np.cos(2*pi*y)]])

# Ackley
#fnc = lambda x, y: -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*np.cos(2*pi*x)+0.5*np.cos(2*pi*y)) + e + 20



Phi = lambda z: fnc(z[0])# just for safety
#DPhi = lambda z: Dfnc(z[0], z[1])

logweightfnc = lambda u: -alpha*Phi(u)

# square_exact = lambda x: fnc(wmean[0],wmean[1]) + Dfnc(wmean[0],wmean[1]).flatten()@(x-wmean) + 0.5*(x-wmean)@(D2fnc(wmean[0][0],wmean[1][0])@(x-wmean))


# global switch: gradient-based yes or no?
use_grad = True
onlymean = True # compute gradient only in mean?

use_truegrad = False # use true gradient instead of inferred Bayesian gradient?
# component_noise = False
# orthogonal_noise = True
noise = "component" #vanilla, component, orthogonal

xmin = -8
xmax = 1.5
ymin = 0
ymax = 80


xs_plot = np.linspace(xmin,xmax,300)
yy = fnc(xs_plot)

d = 1




T = 10.0
tau = 0.01
N = int(T/tau)





# new timing scheme
# r = 0.25
# delta0 = 0.01
# N = 2000
# deltas = [delta0*k**(-r) for k in range(1,N)]
# ts = np.cumsum(deltas)

params = {}
params["alpha"] = 100.0 # weight exponent of cost function exp(-alpha*Phi)
params["alpha_grad"] = 0.0 # weight exponent weighting function used for mean in gradient approximation
params["lam"] = 1.0 # coefficient of contraction towards weighted mean
params["sig"] = 0.5 # coefficient of noise
params["kappa"] = 2.0 # coefficient of gradient drift term
params["avar"] = 100.0

N_ens = 3 # size of ensemble
J = N_ens

N_MC = 100

logweightfnc = lambda u: -alpha*Phi(u)

# inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False)


# initial ensemble
us0_global = np.random.uniform(-8,-1,(d,J)) # hard for functions with minimum at (0,0)
#us0_global = np.array([[-4,-1.5],[-4,-2.5]])
#us0_global = np.random.uniform(-2,2,(2,J)) # hard for Himmelblau


# list of ensemble particles for each iteration
us_list = np.zeros(( d, J, N)) # previously: N, d, j
us_list[:, :, 0] = us0_global

#%%

# plot initial config

plt.figure(figsize=(5,5))
plt.plot(xs_plot, yy)
plt.plot(us_list[0, :, 0], fnc(us_list[0, :, 0]), '.k')


#%%

# first experiment: CBO
params["kappa"] = 0

w_mean = np.zeros((d,N))
w_mean[:, 0] = weighted_mean(us0_global, lambda u: -params["alpha"]*Phi(u))
plt.plot(w_mean[0, 0], fnc(w_mean[0,0]), 'k*', markersize=10)




t1 = time.time()
optimizer = np.zeros((d,N_MC))
for n_MC in range(N_MC):
    # initial ensemble
    # initial ensemble
    us0 = us0_global
    
    # list of ensemble particles for each iteration
    us_list = np.zeros((d, J, N))
    us_list[:, :, 0] = us0
    
    
    
    w_mean = np.zeros((d,N))
    w_mean[:, 0] = weighted_mean(us0, lambda u: -params["alpha"]*Phi(u))
    
    
    
    for i in range(N-1):
        w_mean[:,i] = weighted_mean(us_list[:, :,i], lambda u: -params["alpha"]*Phi(u))
        us_list[:,:,i+1] = step_gradCBO(us_list[:, :, i], Phi, params,tau=tau, maxstep="var", memory=None)
    w_mean[:,N-1]  = weighted_mean(us_list[:, :,N-1], lambda u: -params["alpha"]*Phi(u))
    optimizer[:, n_MC] = w_mean[:, -1]
t2=time.time()
print(f"elapsed time = {t2-t1}")     



fig, ax1 = plt.subplots(1,1, figsize=(5,5))
xdata, ydata = [], []
ln1, = ax1.plot([], [], 'ko')
ln2, = ax1.plot([], [], 'rs')
#plt.plot(xs, ys)
#plt.title("target function")

def init():
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.plot(xs_plot, yy)
    return ln1, ln2,

def update(frame_num): # expects frame to contain the new position of the ensemble
    xdata1 = us_list[0, :, frame_num]
    ydata1 = fnc(xdata1)#us_list[1, :, frame_num]
    ln1.set_data(xdata1, ydata1)
    ln2.set_data(w_mean[0, frame_num], fnc(w_mean[0, frame_num]))
    ax1.set_title(f"t={frame_num}")
    return ln1, 




# f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation.gif" 
# writergif = PillowWriter(fps=30) 
# anim.save(f, writer=writergif)



plt.figure(figsize=(5,5))
ax = plt.gca()
ax2=ax.twinx()
r = 0.25
bins = np.arange(-8-r, 2*r, 2*r)
#bins = [0-r,0+r,1-r,1+r,2-r,2+r,3-r,3+r,4-r,4+r,5-r,5+r]
ax.hist(optimizer.flatten(), bins=bins, density=True, color="orange")
ax2.plot(xs_plot, yy)
ax2.plot(optimizer, fnc(optimizer), 'sk')
    


#%% experiment 2: aug CBO

params["kappa"] = 2.0
params["avar"] = 0


w_mean = np.zeros((d,N))
w_mean[:, 0] = weighted_mean(us0_global, lambda u: -params["alpha"]*Phi(u))




t1 = time.time()
optimizer = np.zeros((d,N_MC))
for n_MC in range(N_MC):
    # initial ensemble
    # initial ensemble
    us0 = us0_global
    
    # list of ensemble particles for each iteration
    us_list = np.zeros((d, J, N))
    us_list[:, :, 0] = us0
    
    
    
    w_mean = np.zeros((d,N))
    w_mean[:, 0] = weighted_mean(us0, lambda u: -params["alpha"]*Phi(u))
    
    
    
    for i in range(N-1):
        w_mean[:,i] = weighted_mean(us_list[:, :,i], lambda u: -params["alpha"]*Phi(u))
        us_list[:,:,i+1] = step_gradCBO(us_list[:, :, i], Phi, params,tau=tau, maxstep="var", memory=None)
    w_mean[:,N-1]  = weighted_mean(us_list[:, :,N-1], lambda u: -params["alpha"]*Phi(u))
    optimizer[:, n_MC] = w_mean[:, -1]
t2=time.time()
print(f"elapsed time = {t2-t1}")     



fig, ax1 = plt.subplots(1,1, figsize=(5,5))
xdata, ydata = [], []
ln1, = ax1.plot([], [], 'ko')
ln2, = ax1.plot([], [], 'rs')
#plt.plot(xs, ys)
#plt.title("target function")

def init():
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.plot(xs_plot, yy)
    return ln1, ln2,

def update(frame_num): # expects frame to contain the new position of the ensemble
    xdata1 = us_list[0, :, frame_num]
    ydata1 = fnc(xdata1)#us_list[1, :, frame_num]
    ln1.set_data(xdata1, ydata1)
    ln2.set_data(w_mean[0, frame_num], fnc(w_mean[0, frame_num]))
    ax1.set_title(f"t={frame_num}")
    return ln1, 




# f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation.gif" 
# writergif = PillowWriter(fps=30) 
# anim.save(f, writer=writergif)



plt.figure(figsize=(5,5))
ax = plt.gca()
ax2=ax.twinx()
r = 0.25
bins = np.arange(-8-r, 2*r, 2*r)
#bins = [0-r,0+r,1-r,1+r,2-r,2+r,3-r,3+r,4-r,4+r,5-r,5+r]
ax.hist(optimizer.flatten(), bins=bins, density=True, color="orange")
ax2.plot(xs_plot, yy)
ax2.plot(optimizer, fnc(optimizer), 'sk')
    

#%% 3rd experiment: aug CBO global

params["kappa"] = 2.0
params["avar"] = 100


w_mean = np.zeros((d,N))
w_mean[:, 0] = weighted_mean(us0_global, lambda u: -params["alpha"]*Phi(u))




t1 = time.time()
optimizer = np.zeros((d,N_MC))
for n_MC in range(N_MC):
    # initial ensemble
    # initial ensemble
    us0 = us0_global
    
    # list of ensemble particles for each iteration
    us_list = np.zeros((d, J, N))
    us_list[:, :, 0] = us0
    
    
    
    w_mean = np.zeros((d,N))
    w_mean[:, 0] = weighted_mean(us0, lambda u: -params["alpha"]*Phi(u))
    
    
    
    for i in range(N-1):
        w_mean[:,i] = weighted_mean(us_list[:, :,i], lambda u: -params["alpha"]*Phi(u))
        us_list[:,:,i+1] = step_gradCBO(us_list[:, :, i], Phi, params,tau=tau, maxstep="var", memory=None)
    w_mean[:,N-1]  = weighted_mean(us_list[:, :,N-1], lambda u: -params["alpha"]*Phi(u))
    optimizer[:, n_MC] = w_mean[:, -1]
t2=time.time()
print(f"elapsed time = {t2-t1}")     



fig, ax1 = plt.subplots(1,1, figsize=(5,5))
xdata, ydata = [], []
ln1, = ax1.plot([], [], 'ko')
ln2, = ax1.plot([], [], 'rs')
#plt.plot(xs, ys)
#plt.title("target function")

def init():
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.plot(xs_plot, yy)
    return ln1, ln2,

def update(frame_num): # expects frame to contain the new position of the ensemble
    xdata1 = us_list[0, :, frame_num]
    ydata1 = fnc(xdata1)#us_list[1, :, frame_num]
    ln1.set_data(xdata1, ydata1)
    ln2.set_data(w_mean[0, frame_num], fnc(w_mean[0, frame_num]))
    ax1.set_title(f"t={frame_num}")
    return ln1, 




# f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation.gif" 
# writergif = PillowWriter(fps=30) 
# anim.save(f, writer=writergif)



plt.figure(figsize=(5,5))
ax = plt.gca()
ax2=ax.twinx()
r = 0.25
bins = np.arange(-8-r, 2*r, 2*r)
#bins = [0-r,0+r,1-r,1+r,2-r,2+r,3-r,3+r,4-r,4+r,5-r,5+r]
ax.hist(optimizer.flatten(), bins=bins, density=True, color="orange")
ax2.plot(xs_plot, yy)
ax2.plot(optimizer, fnc(optimizer), 'sk')

    
ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=1)