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


dim=2  ### CHANGE to 1 for one-dimensional example
    
color_samples = "black"
color_true = "blue"
color_ls = "purple"
color_mean = "green"
   

ind = 0
   

if dim == 2:
    # on a grid
    J = 25 ### CHANGE ensemble size
    xs = np.array([[0,2,0,-2,0],[0,0,2,0,-2]])
    xs = np.array([[8,0],[0,6]])@ np.random.uniform(0,1,(2,50))[:,0:J] - np.array([[4],[4]])
    
    fnc = lambda x, y: np.cos(x) +y-1/36*y**3 + 1/4*x*np.sin(y)
    Dfnc = lambda x, y: (-np.sin(x)+1/4*np.sin(y), 1-1/12*y**2+1/4*x*np.cos(y))
    
    fnc2 = lambda x, y: (x**2+y-11)**2 + (x+y**2-7)**2
    Dfnc2 = lambda x, y: (2*(x**2+y-11)*2*x + 2*(x+y**2-7), 2*(x**2+y-11) + 2*(x+y**2-7)*2*y)
    
    usefnc = fnc2
    useDfnc = Dfnc2
    Phi = lambda u: fnc2(u[0],u[1])
    
    vs = usefnc(xs[0,:], xs[1,:])
    
    xs_plot = np.linspace(-5,5,50)
    ys_plot = np.linspace(-5,5,50)
    XX, YY = np.meshgrid(xs_plot, ys_plot)
    ZZ = usefnc(XX, YY)
    truegrads = useDfnc(xs[0,:], xs[1,:])

d,J = xs.shape

   





hessian = True
sample = False
avar=0.0
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
    sigma_noise = 5.0 # used 1 for 2-d example, 0.2 for 1-d example
    
    prior_mat = sigma_prior**2*np.eye(n_param)
    Kalman_gain = prior_mat @ mat.T @ np.linalg.inv(sigma_noise**2*np.eye(n_obs) + mat@prior_mat@mat.T)
    
    mean_mat[k] = Kalman_gain@vec
    posterior_mat[k] = prior_mat - Kalman_gain@(mat@prior_mat)



for k in range(J):
    ret = inferGradientAndHess(xs, vs, hessian=hessian,ind=k, additionalvariance=avar, scaledversion=scaledversion)
    gk, Hk = ret['grad'], ret['H']
    grads = [*grads, gk]
    HS = [*HS, Hk]

if dim == 2:
    quadratic = lambda x: grad@x + 0.5*x@(H@x)
    plt.figure()#(figsize=(5,5))
    cont = plt.contourf(XX, YY, ZZ, 30, cmap=cm.viridis_r)
    
    plt.colorbar()
    plt.axis("square")
    plt.clim((0,350))
    plt.plot(xs[0,:], xs[1,:], '.')
    if sample:      
        for k in range(J):   
            X = xs-xs[:,k,np.newaxis]     
            n_samples = 5000
            samples = np.random.multivariate_normal(mean_mat[k], posterior_mat[k], n_samples)
            grad_samples = np.stack([coeffToGradHess(samples[s, :], X)[0] for s in range(n_samples)])
            xs_s = np.tile(xs[:,k], (n_samples,1))
            # normalize length
            mean_samples = np.mean(grad_samples, axis=0)
            mean_samples /= np.linalg.norm(mean_samples)
            
            grad_samples /= np.linalg.norm(grad_samples, axis=1)[:,np.newaxis]
            plt.quiver(xs_s[:,0], xs_s[:,1], grad_samples[:,0], grad_samples[:,1], scale_units='inches', scale = 2, headaxislength=0, headlength=0, alpha=0.005)
            #plt.quiver(xs[0,k], xs[1,k], mean_samples[0], mean_samples[1], color=color_mean, scale_units='inches', scale = 3, label="mean")
                
    for k in range(J):
        truegrads = np.stack(truegrads)
        truegrads /= np.linalg.norm(truegrads, axis=0)[np.newaxis,:]
        quiver_true = plt.quiver(xs[0,k], xs[1,k], truegrads[0,k], truegrads[1,k], color=color_true, label="true gradient", scale=3, scale_units="inches")
        if sample:
            patch_samples = mpatches.Patch(color=color_samples, label='gradient samples')
            patch_mean = None#mpatches.Patch(color=color_mean, label='gradient sample mean')
            patch_ls = None
        else:   
            plt.quiver(xs[0,k], xs[1,k], grads[k][0], grads[k][1], color=color_ls, label="max likelihood")
            patch_ls = mpatches.Patch(color=color_ls, label='least squares gradient')
            patch_samples = None
            patch_mean = None
        patch_true = mpatches.Patch(color=color_true, label='true gradient')
        handles=[patch_ls,patch_samples,patch_mean,patch_true]
        plt.legend(handles=[h for h in handles if h is not None])
    
    #alpha=  30
    #wmean = weighted_mean(xs.T, lambda u: -alpha*Phi(u))
    #allpts = np.concatenate((wmean[:,np.newaxis],xs),axis=1)
    #allvs = np.concatenate(([usefnc(wmean[0],wmean[1])],vs))
    #ret_mean = inferGradientAndHess(allpts, allvs, hessian=hessian,ind=0,penalizedistance=penalizedistance, additionalvariance=avar, scaledversion=scaledversion)
    wmean = xs[:,4]
    ret_mean = inferGradientAndHess(xs, vs, hessian=hessian,ind=4, additionalvariance=avar, scaledversion=scaledversion)
    grad_mean, H_mean = ret_mean['grad'], ret_mean['H']
    
    quad_mean = lambda x: usefnc(wmean[0],wmean[1]) + grad_mean@(x-wmean) + 0.5*(x-wmean)@(H_mean@(x-wmean))
    ZZ_quad = np.zeros_like(XX)
    for n, x in enumerate(xs_plot):
        for m, y in enumerate(ys_plot):
            ZZ_quad[m,n] = quad_mean(np.array([x,y]))
    plt.figure();cont2=plt.contourf(XX,YY,ZZ_quad, 30, cmap=cm.viridis_r, levels=cont.levels, extend="both");plt.clim(cont.get_clim())
    plt.colorbar()
    plt.axis("square")
    plt.plot(xs[0,:], xs[1,:], '.k', markersize=5)
    plt.plot(wmean[0], wmean[1], 's', markersize=5)
    #plt.title(f"h={hessian},pd={penalizedistance}")