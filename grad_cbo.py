# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 08:46:58 2022

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, pi
import time


from grad_inference import *

np.random.seed()
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

Phi = lambda u: u**2 + 3*(1-np.cos(2*pi*u))#1/10*(10 + u**2 - 3*np.cos(2*pi*u))
#DPhi = lambda u: 1/10*(2*u+3*2*pi*np.sin(2*pi*u))

# make full Monte Carlo simulation?
MC_test = False
N_MC = 100 # number of Monte Carlo simulations

# global switch: gradient-based yes or no?
use_grad = True
onlymean = True # compute gradient only in mean?

use_truegrad = False # use true gradient instead of inferred Bayesian gradient?

# Parameters
T = 5.0 # time of evolution
N = 500 # number of iterations
tau = T/N # step size

alpha = 50 # weight exponent of cost function exp(-alpha*Phi)
lam = 1.0 # coefficient of contraction towards weighted mean
sig = 0.7 # coefficient of noise
kappa = 1.0 # coefficient of gradient drift term
avar = 1**2
N_ens = 5 # size of ensemble


# inferGradientAndHess(xs, vs, hessian = True, ind=0, penalizedistance=True, retOnlyMatrix=False)


# initial ensemble
sampler = lambda: np.sort(np.random.uniform(3,5,N_ens))
us0 = sampler()

# list of ensemble particles for each iteration
us = np.zeros(( N, len(us0)))
us[0, :] = us0


w_mean = np.zeros((N,1))
w_mean[0,:] = weighted_mean(us0.reshape((-1,1)), lambda u: -alpha*Phi(u))

#%% for plotting
xmin = -1 # for plotting
xmax = np.max(us0)+1
xs = np.linspace(xmin,xmax,500)
ys = Phi(xs)
ymin = np.min(ys)-1
ymax = np.max(ys)+1


#%% 

# method for computing all gradients at once
def compute_gradients(points):
    vs = np.zeros_like(points)
    H = np.zeros((points.shape[0],points.shape[1],points.shape[1]))
    for i in range(points.shape[0]):
        return_dict=inferGradientAndHess(points.T, Phi(points).flatten(), hessian = True, ind=i, penalizedistance=True, additionalvariance=avar)
        vs[i, :], H[i,:,:] = return_dict['grad'], return_dict['H']
    return vs, H



# one step of gradient-augmented CBO
def step_gradHess(us, wmean=None, onlymean=True, threshold_wmean=-np.inf):
    if wmean is None:
        wmean = weighted_mean(us.reshape((-1,1)), lambda u: -alpha*Phi(u))
    #m = weighted_mean(us.reshape((-1,1)), lambda u: -alpha*Phi(u))
    if onlymean:
        if threshold_wmean > -np.inf:
            print("deprecated!")
            distances = (us-wmean)**2
            us_allowed = us[distances > threshold_wmean]
            ensembleplusmean = np.concatenate((wmean,
                                               us_allowed)).reshape((-1,1))
            values = np.concatenate((Phi(wmean),Phi(us_allowed)))            
        else:                
            ensembleplusmean = np.concatenate((wmean,
                                               us)).reshape((-1,1))
            values = np.concatenate((Phi(wmean),Phi(us)))
        return_dict = inferGradientAndHess(ensembleplusmean.T, values, hessian=True, ind=0, penalizedistance=True,additionalvariance=avar)
        vs, H = return_dict['grad'], return_dict['H']
    else:
        vs, H = compute_gradients(us.reshape((-1,1)))
    drift = -tau*kappa*vs.flatten() - lam*tau*(us-wmean)
    return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.abs(us-wmean)*np.random.normal(0,sqrt(tau),(len(us),)), vs, H    

# one step of vanilla CBO
def step(us, wmean):
    drift = - lam*tau*(us-wmean)
    return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.abs(us-wmean)*np.random.normal(0,sqrt(tau),(len(us),))    


def step_truegrad(us, wmean, onlymean=True):
    if onlymean:
        drift = -tau*kappa*DPhi(wmean)- lam*tau*(us-wmean)
        return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.abs(us-wmean)*np.random.normal(0,sqrt(tau),(len(us),))  
    else:
        drift = -tau*kappa*DPhi(us)- lam*tau*(us-wmean)
        return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.abs(us-wmean)*np.random.normal(0,sqrt(tau),(len(us),))  
    


if MC_test:    
    
    t1 = time.time()
    optimizer = np.zeros((N_MC))
    for n_MC in range(N_MC):
        # initial ensemble
        us0 = sampler()
        
        # list of ensemble particles for each iteration
        us = np.zeros(( N, len(us0)))
        us[0, :] = us0
        
        
        w_mean = np.zeros((N))
        w_mean[0] = weighted_mean(us0.reshape((-1,1)), lambda u: -alpha*Phi(u))
        
        
        for i in range(N-1):
            #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
            m = weighted_mean(us[i,:].reshape((-1,1)), lambda u: -alpha*Phi(u))
            w_mean[i+1] = m
            if use_grad:
                if use_truegrad:
                    us[i+1, :] = step_truegrad(us[i, :],m,onlymean=onlymean)
                else:
                    us[i+1, :], *args = step_gradHess(us[i, :],m,onlymean=onlymean, threshold_wmean=-np.inf)
            else:    
                us[i+1, :] = step(us[i, :],m)
        optimizer[n_MC] = w_mean[-1]
    t2=time.time()
    print(f"elapsed time = {t2-t1}")        
    
    fig, ax1 = plt.subplots(1,1)
    xdata, ydata = [], []
    ln1, = ax1.plot([], [], 'ko')
    ln2, = ax1.plot([], [], 'rs')
    
    def init():
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.plot(xs,ys)
        return ln1, ln2,
    
    def update(frame_num): # expects frame to contain the new position of the ensemble
        xdata1 = us[frame_num, :]
        ydata1 = Phi(xdata1)
        ln1.set_data(xdata1, ydata1)
        ln2.set_data(w_mean[frame_num], Phi(w_mean[frame_num]))
        ax1.set_title(f"it={frame_num}")
        return ln1, 
    ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=50)
    
    xmin = -1 # for plotting
    xmax = 5
    xs = np.linspace(xmin,xmax,500)
    ys = Phi(xs)
    ymin = np.min(ys)-1
    ymax = np.max(ys)+1
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(us, 'k')
    plt.plot(w_mean, 'r')
    plt.title("position of particles")
    plt.subplot(2,1,2)
    plt.plot(Phi(us), 'k')
    plt.plot(Phi(w_mean), 'r')
    plt.title("value of particles")
    plt.tight_layout()
    
    plt.figure()
    #plt.plot(xs, np.exp(-alpha*Phi(xs)), 'k', alpha=0.5)
    plt.plot(xs,ys)
    plt.plot(optimizer,Phi(optimizer), 'k.')
    #plt.hist(optimizer)
    plt.tight_layout()
    
    r = 0.1
    bins = np.arange(-r, 5+2*r, 2*r)
    #bins = [0-r,0+r,1-r,1+r,2-r,2+r,3-r,3+r,4-r,4+r,5-r,5+r]
    plt.hist(optimizer, bins=bins, density=True)
    
    # if use_grad:
    #     if use_truegrad:
    #         if onlymean:        
    #             string_save="img/statistics_CBO_aug_truegrad_onlymean.pdf"
    #         else:   
    #             string_save="img/statistics_CBO_aug_truegrad.pdf"
    #     else:
    #         if onlymean:        
    #             string_save="img/statistics_CBO_aug_onlymean.pdf"
    #         else:   
    #             string_save="img/statistics_CBO_aug.pdf"
    # else:
    #     string_save="img/statistics_CBO.pdf"
    # plt.savefig(string_save, dpi=500)


else: # only one (animated) run instead of full MC simulation
    #%% prepare animation
    fig, ax1 = plt.subplots(1,1)
    xdata, ydata = [], []
    ln1, = ax1.plot([], [], 'ko')
    ln2, = ax1.plot([], [], 'rs')
    ln3, = ax1.plot([],[],'b-')
    #plt.plot(xs, ys)
    #plt.title("target function")
    
    list_grads = [None for i in range(N-1)]
    list_Hs = [None for i in range(N-1)]
    def init():
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.plot(xs,ys)
        return ln1, ln2, ln3,
    
    def update(frame_num): # expects frame to contain the new position of the ensemble
        xdata1 = us[frame_num, :]
        ydata1 = Phi(xdata1)
        ln1.set_data(xdata1, ydata1)
        ln2.set_data(w_mean[frame_num], Phi(w_mean[frame_num]))
        fnc_interpolant = lambda x: (Phi(w_mean[frame_num]) + list_grads[frame_num]*(x-w_mean[frame_num]) + 0.5*(x-w_mean[frame_num])*(list_Hs[frame_num]*(x-w_mean[frame_num])))[0]
        ln3.set_data(xs, fnc_interpolant(xs))
        ax1.set_title(f"it={frame_num}")
        return ln1, ln2, ln3
    
    #%% do iteration
    for i in range(N-1):
        #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
        m = weighted_mean(us[i,:].reshape((-1,1)), lambda u: -alpha*Phi(u))
        w_mean[i+1,:] = m
        if use_grad:
            if use_truegrad:
                us[i+1, :] = step_truegrad(us[i, :],m,onlymean=onlymean)
            else:
                us[i+1, :], list_grads[i], list_Hs[i] = step_gradHess(us[i, :],m,onlymean=onlymean)
        else:    
            us[i+1, :] = step(us[i, :],m)
        
    ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=50)
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(us, 'k')
    plt.plot(w_mean, 'r')
    plt.title("position of particles")
    plt.subplot(2,1,2)
    plt.plot(Phi(us), 'k')
    plt.plot(Phi(w_mean), 'r')
    plt.title("value of particles")
    plt.tight_layout()
