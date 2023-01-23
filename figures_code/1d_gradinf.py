# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:11:35 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import logsumexp
from scipy.sparse.linalg import lsmr
from matplotlib import cm
import matplotlib.patches as mpatches

from math import pi



from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from grad_inference import *

np.random.seed(6)


use_lsmr = True
sig_global = 0.5


dim=1  ### CHANGE to 1 for one-dimensional example
    
color_samples = "black"
color_true = "blue"
color_ls = "purple"
color_mean = "green"
   

ind = 0
if dim == 1:
    xs = np.array([[0,1,-2]])
    vs = np.array([0,1,4])
    #fnc = lambda x: x**2
    #usefnc = fnc
    J = 7
    
    xs = np.concatenate((np.random.uniform(-2,1,J-2), np.array([2.5]), np.array([3]))).reshape((1,-1))
    usefnc = lambda u: u**2 + 3*(1-np.cos(2*pi*u))#1/10*(10 + u**2 - 3*np.cos(2*pi*u))
    Dusefnc = lambda u: 2*u**2 + 3*2*pi*(np.sin(2*pi*u))#1/10*(10 + u**2 - 3*np.cos(2*pi*u))
    
    vs = usefnc(xs).flatten()
    
    V = vs-vs[ind]
    X = xs-xs[:,ind].reshape((-1,1))
    truegrads = Dusefnc(xs).flatten()
    
       
    d,J = xs.shape
# 1d example
    sample_toggle = 0
    for avar in [0,0,1,1000]:
        
        hessian = True
        sample = False
        if sample_toggle > 0 and avar == 0:
            sample = True
        sample_toggle += 1
        scaledversion = True
        ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=0, additionalvariance=avar, scaledversion=scaledversion)
        grad, H = ret['grad'], ret['H']
        grads = []
        HS = []
        
        
        mean_mat = [None for j in range(J)]
        posterior_mat = [None for j in range(J)]
        
        for k in range(J):
            ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=k, retOnlyMatrix=True, additionalvariance=avar, scaledversion=scaledversion)
            mat, vec = ret['A'], ret['b']
            
            n_param = mat.shape[1]
            n_obs = mat.shape[0]
            
            sigma_prior = 2
            sigma_noise = 100.0 # used 1 for 2-d example, 0.2 for 1-d example
            
            prior_mat = sigma_prior**2*np.eye(n_param)
            Kalman_gain = prior_mat @ mat.T @ np.linalg.inv(sigma_noise**2*np.eye(n_obs) + mat@prior_mat@mat.T)
            
            mean_mat[k] = Kalman_gain@vec
            posterior_mat[k] = prior_mat - Kalman_gain@(mat@prior_mat)
    
    
    
        for k in range(J):
            ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=k, additionalvariance=avar, scaledversion=scaledversion)
            gk, Hk = ret['grad'], ret['H']
            grads = [*grads, gk]
            HS = [*HS, Hk]
    
    
        plt.figure()
        xs_plot = np.linspace(np.min(xs)-0.5, np.max(xs)+0.5,200)
        plt.plot(xs_plot, usefnc(xs_plot), 'k', alpha=0.5, linewidth=1.0)
        plt.plot(xs.flatten(), vs, '.')
        #plt.plot(xs[:,ind], vs[ind], 'rs')
        
        for k in range(J):       
            
            if sample:
                n_samples = 600
                samples = np.random.multivariate_normal(mean_mat[k], posterior_mat[k], n_samples)
                samples_grad = samples[:,0:J]
                samples_lambda = samples[:, J:]
                samples_hess = [None for m in range(n_samples)]
                X = xs-xs[:,k].reshape((-1,1))
                norms_X = np.linalg.norm(X, axis=0, keepdims=True)
                Z = np.copy(X)
                if scaledversion:
                    Z = np.divide(X, norms_X, out=Z, where=(norms_X != 0))
                norms = (np.linalg.norm(X, axis=0)).reshape((1,-1))
                for s in range(n_samples):    
                    lambdas = samples_lambda[s, :]
                    H_sample = np.zeros((d,d))
                    for l in range(len(lambdas)):
                        if l != k:
                            xx = X[:,l].reshape((-1,1))
                            H_sample += lambdas[l]*(xx@xx.T)/(norms[0,l])**2
                    samples_hess[s] = H_sample
            
                mean_grad = [coeffToGradHess(mean_mat[k], Z)[0] for k in range(J)]#####[coeffToGradHess(xs, k, mean_mat[k])[0] for k in range(J)]
                mean_hess = [coeffToGradHess(mean_mat[k], Z)[1] for k in range(J)]#####[coeffToGradHess(xs, k, mean_mat[k])[1] for k in range(J)]
                # plot samples
                for s in range(n_samples):
                    grad_sample, Hess_sample = coeffToGradHess(samples[s, :], Z)#####coeffToGradHess(xs, k, samples[s, :])                
                    if H is None:
                        fnc_sample =  lambda x: (vs[k] + (grad_sample)*(x-xs[:,k]))[0]
                    else:
                        fnc_sample = lambda x: (vs[k] + grad_sample*(x-xs[:,k]) + 0.5*(x-xs[:,k])@(Hess_sample@(x-xs[:,k])))[0]
                    sample_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [fnc_sample(xs_ploti) for xs_ploti in xs_plot])
                    plt.plot(xs_plot, sample_vals, 'k', alpha=0.01)
                sample_mean = np.mean(samples, axis=0)
                
                grad_sample_mean, Hess_sample_mean  = coeffToGradHess(sample_mean, Z)
                if H is None:
                    fnc_sample =  lambda x: (vs[k] + (grad_sample_mean)*(x-xs[:,k]))[0]
                else:
                    fnc_sample = lambda x: (vs[k] + grad_sample_mean*(x-xs[:,k]) + 0.5*(x-xs[:,k])@(Hess_sample_mean@(x-xs[:,k])))[0]
                sample_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [fnc_sample(xs_ploti) for xs_ploti in xs_plot])
                plt.plot(xs_plot, sample_vals, color=color_mean)
                patch_samples = mpatches.Patch(color=color_samples, label='gradient samples')
                patch_mean = mpatches.Patch(color=color_mean, label='gradient sample mean')
            else:
                patch_samples = None
                patch_mean = None
            if H is None:
                quadratic = lambda x: (vs[k] + grads[k]*(x-xs[:,k]))[0]
            else:
                quadratic = lambda x: (vs[k] + grads[k]*(x-xs[:,k]) + 0.5*(x-xs[:,k])@(HS[k]@(x-xs[:,k])))[0]
            quad_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [quadratic(xs_ploti) for xs_ploti in xs_plot])
            trueTangent = lambda x: (vs[k] + truegrads[k]*(x-xs[:,k]))[0]
            
            trueD_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [trueTangent(xs_ploti) for xs_ploti in xs_plot])
            plt.plot(xs_plot, quad_vals, color=color_ls, alpha=0.5)
            
            patch_ls = mpatches.Patch(color=color_ls, label='least squares gradient')
            patch_true = None# mpatches.Patch(color=color_true, label='true gradient')
            handles=[patch_ls,patch_samples,patch_mean,patch_true]
            #plt.legend([h for h in handles if h is not None])
            
            plt.legend(handles=[h for h in handles if h is not None], loc="upper left")
            
            plt.ylim([0,15])
        
        #plt.title(f"H={hessian}, PD={penalizedistance}")
        
        
        plt.show()
    
    # last example: Extrapolation! first a small, then a large
    avar = 0
    ind = 0 # index of extrapolation
    hessian = True
    sample = False
    scaledversion = True
    ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=0, additionalvariance=avar, scaledversion=scaledversion)
    grad, H = ret['grad'], ret['H']
    HS = []
    
    X = xs-xs[:,ind].reshape((-1,1))
    grads = grad + H@X
    
    
    
    plt.figure()
    xs_plot = np.linspace(np.min(xs)-0.5, np.max(xs)+0.5,200)
    plt.plot(xs_plot, usefnc(xs_plot), 'k', alpha=0.5, linewidth=1.0)
    plt.plot(xs.flatten(), vs, '.')
    #plt.plot(xs[:,ind], vs[ind], 'rs')
    
    for k in range(J):       
        
        
        patch_samples = None
        patch_mean = None
        if H is None:
            quadratic = lambda x: (vs[k] + grads[:,k]*(x-xs[:,k]))[0]
        else:
            quadratic = lambda x: (vs[k] + grads[:,k]*(x-xs[:,k]) + 0.5*(x-xs[:,k])@(H@(x-xs[:,k])))[0]
        quad_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [quadratic(xs_ploti) for xs_ploti in xs_plot])
        trueTangent = lambda x: (vs[k] + truegrads[k]*(x-xs[:,k]))[0]
        
        trueD_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [trueTangent(xs_ploti) for xs_ploti in xs_plot])
        plt.plot(xs_plot, quad_vals, color=color_ls, alpha=0.5)
        
        patch_ls = mpatches.Patch(color=color_ls, label='least squares gradient')
        patch_true = None# mpatches.Patch(color=color_true, label='true gradient')
        handles=[patch_ls,patch_samples,patch_mean,patch_true]
        #plt.legend([h for h in handles if h is not None])
        
        plt.legend(handles=[h for h in handles if h is not None], loc="upper left")
        
        plt.ylim([0,15])
        
        
    avar = 1000
    ind = 0 # index of extrapolation
    hessian = True
    sample = False
    scaledversion = True
    ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=0, additionalvariance=avar, scaledversion=scaledversion)
    grad, H = ret['grad'], ret['H']
    HS = []
    
    X = xs-xs[:,ind].reshape((-1,1))
    grads = grad + H@X
    
    
    
    plt.figure()
    xs_plot = np.linspace(np.min(xs)-0.5, np.max(xs)+0.5,200)
    plt.plot(xs_plot, usefnc(xs_plot), 'k', alpha=0.5, linewidth=1.0)
    plt.plot(xs.flatten(), vs, '.')
    #plt.plot(xs[:,ind], vs[ind], 'rs')
    
    for k in range(J):       
        
        
        patch_samples = None
        patch_mean = None
        if H is None:
            quadratic = lambda x: (vs[k] + grads[:,k]*(x-xs[:,k]))[0]
        else:
            quadratic = lambda x: (vs[k] + grads[:,k]*(x-xs[:,k]) + 0.5*(x-xs[:,k])@(H@(x-xs[:,k])))[0]
        quad_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [quadratic(xs_ploti) for xs_ploti in xs_plot])
        trueTangent = lambda x: (vs[k] + truegrads[k]*(x-xs[:,k]))[0]
        
        trueD_vals = np.ma.masked_where(np.abs(xs_plot-xs[:,k]) > 0.5, [trueTangent(xs_ploti) for xs_ploti in xs_plot])
        plt.plot(xs_plot, quad_vals, color=color_ls, alpha=0.5)
        
        patch_ls = mpatches.Patch(color=color_ls, label='least squares gradient')
        patch_true = None# mpatches.Patch(color=color_true, label='true gradient')
        handles=[patch_ls,patch_samples,patch_mean,patch_true]
        #plt.legend([h for h in handles if h is not None])
        
        plt.legend(handles=[h for h in handles if h is not None], loc="upper left")
        
        plt.ylim([0,15])
    
    #plt.title(f"H={hessian}, PD={penalizedistance}")
    
    
    plt.show()