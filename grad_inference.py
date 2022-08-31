
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:33:17 2022

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

np.random.seed(6)


use_lsmr = True
sig_global = 0.5

matplotlib.rcParams.update({'font.size': 13})

def coeffToGradHess(coeffs, X):
    
    d,J = X.shape
    
    if len(coeffs) == 2*J:
        grad = X @ coeffs[0:J]
        coeffs_Hess = coeffs[J:]
        H = np.einsum('i,ki,li->kl', coeffs_Hess, X, X)#coeffs[J:]*Z@X.T
        return grad, H
    else:
        grad = X @ coeffs
        return grad, None


def inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False, coeffs0=None, additionalvariance=0.0, scaledversion=False):    
    d,J = xs.shape
    X = xs-xs[:,ind,np.newaxis]
    V = vs-vs[ind]
    norms_X = np.linalg.norm(X, axis=0, keepdims=True)
    Z = np.copy(X)
    if scaledversion:
        Z = np.divide(X, norms_X, out=Z, where=(norms_X != 0))
    mat1 = X.T@Z
    return_dict = {'grad': None, 'H': None, 'coeffs_grad': None, 'coeffs_H': None, 'A': None, 'b': None}
    if hessian:
        mat2 = (mat1**2)/2
        mat = np.concatenate((mat1, mat2), axis=1)
        
        b = np.zeros_like(V,dtype=float)
        divisor_b = sig_global**2*norms_X.flatten()**3 +additionalvariance#+ sig_global**2*norms_X.flatten()**4 
        b = np.divide(V, divisor_b, out=b, where=(divisor_b!=0))
        
        return_dict['b'] = b
        
        A = np.zeros_like(mat,dtype=float)
        divisor_A = np.tile(sig_global**2*(norms_X.T**3),(1,2*J))  + additionalvariance
        #divisor_A = np.tile(sig_global**2*(norms_X.T**3+norms_X.T**4),(1,2*J))  + additionalvariance
        A = np.divide(mat, divisor_A, out=A, where=(divisor_A!=0))
        
        return_dict['A'] = A
        #print(f"matrix: size={A.shape}")
        if retOnlyMatrix:
            return return_dict
        else:
            if use_lsmr:
                coeffs_lsq = lsmr(A, b, x0=coeffs0)[0]
            else:
                #print("warning, this is slow")
                coeffs_lsq = np.linalg.lstsq(A, b, rcond=None)[0]
            return_dict['coeffs_grad'] = coeffs_lsq[0:J]
            return_dict['coeffs_H'] = coeffs_lsq[J:]
            return_dict['coeffs'] = coeffs_lsq
            #return_dict['grad'], return_dict['H'] = coeffToGradHess(xs, ind, coeffs_lsq)#
            return_dict['grad'], return_dict['H'] = coeffToGradHess(coeffs_lsq, Z) ##########coeffToGradHess(coeffs_lsq, X)
            return return_dict
    else:        
        b = np.zeros_like(V,dtype=float)
        divisor_b = sig_global**2*norms_X.flatten()**3+ additionalvariance
        b = np.divide(V, divisor_b, out=b, where=(divisor_b!=0))
        
        return_dict['b'] = b
        
        A_short = np.zeros_like(mat1,dtype=float)
        divisor_mat1 = np.tile(norms_X.T**3,(1,J))
        A_short = np.divide(mat1, divisor_mat1, out=A_short, where=(divisor_mat1!=0))
        
        return_dict['A'] = A_short
        
        if retOnlyMatrix:
            return return_dict
        else:
            if use_lsmr:
                coeffs_lsq = lsmr(A_short, b)[0]
            else:
                #print("warning, this is slow")
                coeffs_lsq = np.linalg.lstsq(A_short, b, rcond=None)[0]
            return_dict['coeffs_grad'] = coeffs_lsq
            return_dict['grad'], return_dict['H'] = coeffToGradHess(coeffs_lsq, Z) ##########coeffToGradHess(coeffs_lsq, X)
            return return_dict
            
        
    



def weighted_mean(xs, logweightfnc, allow_vectorized_input=True):
    # assume: xs.shape = (d,J)
    if allow_vectorized_input:
        logweights = logweightfnc(xs)#np.array([logweightfnc(xs[j, :]) for j in range(xs.shape[0])])
    else:
        logweights = np.array([logweightfnc(xs[:,j]) for j in range(xs.shape[1])])
    logsumweights = logsumexp(logweights)
    logweights = logweights - logsumweights
    weights = np.exp(logweights)
    return xs@weights

if __name__ == "__main__":
    
    dim=2  ### CHANGE to 1 for one-dimensional example
    
    color_samples = "black"
    color_true = "blue"
    color_ls = "purple"
    color_mean = "green"
   
    
    ind = 0
    if dim == 1:
    # 1d example
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
        
       
    
    if dim == 2:
        # on a grid
        J = 5 ### CHANGE ensemble size
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
    sample = True
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
        
        sigma_prior = 15
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
    
    if dim == 1:
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
    
    if dim == 2:
        quadratic = lambda x: grad@x + 0.5*x@(H@x)
        fig = plt.figure()#(figsize=(5,5))
        ax0 = plt.gca()
        cont = plt.contourf(XX, YY, ZZ, 30, cmap=cm.viridis_r)
        
        plt.colorbar()
        plt.axis("square")
        plt.clim((0,350))
        plt.plot(xs[0,:], xs[1,:], '.')
        fig_rad = plt.figure()
        ax_rad = plt.gca()
        if sample:      
            for k in range(J):   
                truegrads = np.stack(truegrads)
                truegrads /= np.linalg.norm(truegrads, axis=0)[np.newaxis,:]
                X = xs-xs[:,k,np.newaxis]     
                n_samples = 5000
                samples = np.random.multivariate_normal(mean_mat[k], posterior_mat[k], n_samples)
                grad_samples = np.stack([coeffToGradHess(samples[s, :], X)[0] for s in range(n_samples)])
                xs_s = np.tile(xs[:,k], (n_samples,1))
                # normalize length
                mean_samples = np.mean(grad_samples, axis=0)
                mean_samples /= np.linalg.norm(mean_samples)
                
                grad_samples /= np.linalg.norm(grad_samples, axis=1)[:,np.newaxis]
                angles = np.array([np.arctan2(grad_samples[n, 1], grad_samples[n,0]) for n in range(n_samples)])
                ax0.quiver(xs_s[:,0], xs_s[:,1], grad_samples[:,0], grad_samples[:,1], scale_units='inches', scale = 2, headaxislength=0, headlength=0, alpha=0.005)
                #plt.quiver(xs[0,k], xs[1,k], mean_samples[0], mean_samples[1], color=color_mean, scale_units='inches', scale = 3, label="mean")
                  
                bottom = 2
                max_height = 6
                n_in_bins, bins= np.histogram(angles, bins=np.linspace(-2*pi,2*pi,300))
                radii = max_height*n_in_bins/max(n_in_bins)
                thetas = (bins[0:-1]+bins[1:])/2
                width = (2*np.pi) / 150
                #plt.figure()
                if k < 4:
                    print(k)
                    plt.figure()
                    ax = plt.subplot(111, polar=True)#plt.subplot(2,2,k+1, polar=True)
                    #ax = plt.subplot(111, polar=True)
                    bars = ax.bar(thetas, radii, width=width, bottom=bottom)
                    angle_true = np.arctan2(np.array([truegrads[1,k]]), np.array([truegrads[0,k]]))
                    angles_true = np.tile(angle_true,2)
                    radii_mean = max_height*np.array([0,1])
                    # Use custom colors and opacity
                    for r, bar in zip(radii, bars):
                        bar.set_facecolor(plt.cm.viridis_r(r/2))
                        #bar.set_alpha()
                    #ax.plot(angles_true, radii_mean, linewidth=5)
                    ax.set_rticks([]) 
                    ax.set_rmax(3)
                    ax.grid(True)
                    ax.set_thetagrids([0,45,90,135,180,225,270,315],[])
                    ax.arrow(angle_true[0],0,0,3.5, head_width=0.2, width=0.05, head_length=1.5)
                    #plt.title(str(xs[:,k]))
                    plt.tight_layout()
                
                    
        for k in range(J):
            quiver_true = ax0.quiver(xs[0,k], xs[1,k], truegrads[0,k], truegrads[1,k], color=color_true, label="true gradient", scale=3, scale_units="inches")
            if sample:
                patch_samples = mpatches.Patch(color=color_samples, label='gradient samples')
                patch_mean = None#mpatches.Patch(color=color_mean, label='gradient sample mean')
                patch_ls = None
            else:   
                ax0.quiver(xs[0,k], xs[1,k], grads[k][0], grads[k][1], color=color_ls, label="max likelihood")
                patch_ls = mpatches.Patch(color=color_ls, label='least squares gradient')
                patch_samples = None
                patch_mean = None
            patch_true = mpatches.Patch(color=color_true, label='true gradient')
            handles=[patch_ls,patch_samples,patch_mean,patch_true]
            ax0.legend(handles=[h for h in handles if h is not None])
        
      
        
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
            
            
            
