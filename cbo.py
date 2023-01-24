# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:58:51 2022

@author: Philipp


NEW FILE FOR PUBLICATION AND FIGURE GENERATION
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, e, exp
import time



from grad_inference import *
from matplotlib.animation import FuncAnimation, PillowWriter

exponential_integrator = False  # toggle whether to use exponential integrators \
                                # for numerical scheme (otherwise Euler-Maruyama)

# one step of gradient-augmented CBO
def step_gradCBO(us, fnc, params, tau=0.01, noise="vanilla", maxstep=np.inf, memory=None, allow_vectorized_input=True, extrapolate_grad=False):
    
    d,J = us.shape
    wmean_grad =  weighted_mean(us, lambda u: -params["alpha_grad"]*fnc(u),allow_vectorized_input=allow_vectorized_input)[:,np.newaxis]
    wmean = weighted_mean(us, lambda u: -params["alpha"]*fnc(u),allow_vectorized_input=allow_vectorized_input)[:,np.newaxis]
    if params["kappa"] > 0:
        if memory is not None:
            ensembleplusmean = np.concatenate((wmean_grad,us,memory),axis=1)
        else:
            ensembleplusmean = np.concatenate((wmean_grad,us),axis=1)
            
        if allow_vectorized_input:
            values = fnc(ensembleplusmean)
        else:
            values = np.array([fnc(ensembleplusmean[:,k]) for k in range(J+1)])
        return_dict= inferGradientAndHess(ensembleplusmean, values, hessian=True, ind=0, additionalvariance=params["avar"])
        vs, H =  return_dict['grad'], return_dict['H']
        #vs = np.linalg.solve(H + np.eye(d), vs) # experimental to avoid oscillations
        
        if maxstep == "var": # if keyword is "var", then adapt tau according to "variance" of ensemble
            var = np.linalg.norm(us-wmean)        
            gradnorm = np.linalg.norm(vs)
            tau = min(tau, 0.1*var*params["lam"]/(gradnorm*params["kappa"]))
        else:
            if maxstep == None or maxstep == np.inf:
                pass
            else:
                tau = max(tau, maxstep/(params["kappa"]*np.linalg.norm(vs))) # adapt step such that -tau*kappa*vs is less than maxstep
            #print(tau)
        if extrapolate_grad == True:
            vs = vs[:, np.newaxis] + H@(us - wmean)
        else:
            vs = vs[:, np.newaxis]
    else: 
        vs = np.zeros(us.shape)
    if noise == "vanilla":
        diff = params["sig"]*np.linalg.norm(us-wmean, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
    elif noise == "component":
        diff = params["sig"]*(us-wmean)*np.random.normal(0,sqrt(tau),(d,J))    
    elif noise =="orthogonal":
        diff0 = params["sig"]*np.linalg.norm(us-wmean, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
        us_step = us + 0.5*diff0
        m_step = weighted_mean(us_step, lambda u: -params["alpha"]*fnc(u),allow_vectorized_input=allow_vectorized_input)[:,np.newaxis]
        
        term1 = (np.einsum("ij,ij->j", (us_step-m_step), diff0))#term1 = (np.einsum("ij,ij->i", (us_step-m_step), diff0))
        term2 = (np.einsum("ij,ij->j", (us_step-m_step), (us_step-m_step)))#term2 = (np.einsum("ij,ij->i", (us_step-m_step), (us_step-m_step)))
        division = np.divide(term1, term2, where=term2 > 0)
        diff = diff0 - division*(us_step-m_step)
        
    drift = -tau*params["kappa"]*vs - params["lam"]*tau*(us-wmean)#/np.linalg.norm(us-wmean)**(1/2)
    #print(tau)
    if exponential_integrator:
        pass
    else:
        return us + drift + diff