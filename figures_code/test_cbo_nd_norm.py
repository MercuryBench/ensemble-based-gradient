# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:31:16 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse.linalg import lsmr
from math import sqrt, pi, e, sqrt, ceil, exp
import time

from grad_inference import *
from cbo import *
from matplotlib.animation import FuncAnimation, PillowWriter


np.random.seed(1)
d = 10
 


# squared norm

Phi = lambda z: 0.5*np.linalg.norm(z-1, axis=0)**2
DPhi = lambda z: z




# square_exact = lambda x: fnc(wmean[0],wmean[1]) + Dfnc(wmean[0],wmean[1]).flatten()@(x-wmean) + 0.5*(x-wmean)@(D2fnc(wmean[0][0],wmean[1][0])@(x-wmean))






logweightfnc = lambda u: -alpha*Phi(u)

# square_exact = lambda x: fnc(wmean[0],wmean[1]) + Dfnc(wmean[0],wmean[1]).flatten()@(x-wmean) + 0.5*(x-wmean)@(D2fnc(wmean[0][0],wmean[1][0])@(x-wmean))


# global switch: gradient-based yes or no?
use_grad = True
onlymean = True # compute gradient only in mean?

use_truegrad = False # use true gradient instead of inferred Bayesian gradient?
component_noise = True
orthogonal_noise = False



T = 60.0
tau = 0.02
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
params["sig"] = 0.2 # coefficient of noise
params["kappa"] = 4.0 # coefficient of gradient drift term
params["avar"] = 0.0

N_ens = 20 # size of ensemble
J = N_ens


logweightfnc = lambda u: -alpha*Phi(u)

# inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False)


# initial ensemble
us0_global = np.random.uniform(-4,-1,(d,J)) # hard for functions with minimum at (0,0)
#us0_global = np.array([[-4,-1.5],[-4,-2.5]])
#us0_global = np.random.uniform(-2,2,(2,J)) # hard for Himmelblau

for kap in [0, 4.0]:
    for N_ens in [5,20]:
        J = N_ens
        params["kappa"] = kap
                
        # list of ensemble particles for each iteration
        us_list = np.zeros(( d, J, N)) # previously: N, d, j
        us_list[:, :, 0] = us0_global[:, 0:J]
        w_mean = np.zeros((d,N))
        w_mean[:, 0] = weighted_mean(us0_global, lambda u: -params["alpha"]*Phi(u))
        #%%
        
        
        
        for i in range(N-1):
            #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
            w_mean[:,i] = weighted_mean(us_list[:, :,i], lambda u: -params["alpha"]*Phi(u))
            us_list[:,:,i+1] = step_gradCBO(us_list[:, :, i], Phi, params,tau=tau, maxstep=None, noise="component")
            
        w_mean[:,N-1]  = weighted_mean(us_list[:, :,N-1], lambda u: -params["alpha"]*Phi(u))
            
        
        
        # f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation.gif" 
        # writergif = PillowWriter(fps=30) 
        # anim.save(f, writer=writergif)
        
        
        
        plt.figure(figsize=(5,3))
        #plt.subplot(2,1,1)
        #plt.title("weighted mean")
        # it = np.linspace(0.0, T, N)
        
        # ns = np.arange(N)
        # ts = np.array([it[n] for n in ns])
        plt.semilogy([Phi(w_mean[:,n]) for n in range(N)], label="val of weighted mean")
        vals = np.log(np.array([Phi(w_mean[:,n]) for n in range(N)]))
        Phi_mat = np.stack((range(N), np.ones(N))).T
        plt.tight_layout()
        #coeff = np.linalg.lstsq(Phi_mat, vals, rcond=None)[0]
           
        
        
          
        # #plt.semilogy(np.exp(coeff[1] + coeff[0]*np.arange(N)), label=f"$\exp({coeff[1]:.5f} {coeff[0]:.5f}\cdot n)$")
        # plt.legend()
        
        plt.figure(figsize=(5,3))
        plt.loglog([Phi(w_mean[:,n]) for n in range(N)], label="val of weighted mean")
        #plt.plot(np.exp(coeff[1] + coeff[0]*np.arange(N)), label=f"$\exp({coeff[1]:.5f} {coeff[0]:.5f}\cdot n)$")
        plt.tight_layout()
        
        

        
        
        
        

            



"""

T = 20.0


# tau = 0.001
# N = int(T/tau)
# deltas = [tau for k in range(1,N)]
# ts = np.arange(0,T,tau)


# new timing scheme
r = 0.3
delta0 = 0.01
N = 5000
deltas = [delta0*k**(-r) for k in range(1,N+1)]
ts = np.cumsum(deltas)

tau0 = delta0

# Parameter für Rastrigin 50-d
alpha = 30.0 # weight exponent of cost function exp(-alpha*Phi)
lam = 1.5 # coefficient of contraction towards weighted mean
sig = 0.7 # coefficient of noise
kappa = 0.5 # coefficient of gradient drift term
N_ens = 40 # size of ensemble
J = N_ens
avar = 0.0

# Parameter für quadratische Funktion 50-d
# alpha = 30.0 # weight exponent of cost function exp(-alpha*Phi)
# lam = 1.0 # coefficient of contraction towards weighted mean
# sig = 4.0 # coefficient of noise 0.2
# kappa = 0.5#2.5#1.5 # coefficient of gradient drift term
# N_ens = 10 # size of ensemble
# J = N_ens
# avar = 0.0

# inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False)


# initial ensemble
us0_global = np.random.uniform(-3,3,(d,J)) # hard for functions with minimum at (0,0)
#us0_global = np.random.uniform(-2,2,(2,J)) # hard for Himmelblau


# list of ensemble particles for each iteration
us_list = np.zeros(( N, d, J))
us_list[0, :, :] = us0_global

#%%

# plot initial config

# plt.figure(figsize=(5,5))
# plt.contourf(XX, YY, ZZ, 30)
# plt.plot(us_list[0, 0, :], us_list[0, 1, :], '.k')

w_mean = np.zeros((N,len(us0_global)))
w_mean[0, :] = weighted_mean(us0_global.T, lambda u: -alpha*Phi(u))
# plt.plot(w_mean[0, 0], w_mean[0,1], 'rx')



# method for computing all gradients at once
def compute_gradients(points):
    vs = np.zeros_like(points)
    H = np.zeros((points.shape[0],points.shape[1],points.shape[1]))
    for i in range(points.shape[0]):
        return_dict= inferGradientAndHess(points.T, Phi(points.T).flatten(), hessian = True, ind=i, additionalvariance=avar)
        vs[i, :], H[i,:,:] = return_dict['grad'], return_dict['H']
    return vs, H



# one step of gradient-augmented CBO
def step_gradHess(us, m, onlymean=True, tau=1.0, it=0, coeffs0=None, memory=None):
    #m = weighted_mean(us_list.reshape((-1,1)), lambda u: -alpha*Phi(u))
    if onlymean:
        if memory is not None:
            ensembleplusmean = np.concatenate((m,us,memory),axis=1)
        else:
            ensembleplusmean = np.concatenate((m,us),axis=1)            
        values = Phi(ensembleplusmean)
        #values = np.concatenate((Phi(m),Phi(us)))
        return_dict= inferGradientAndHess(ensembleplusmean, values, hessian=True, ind=0, additionalvariance=avar, coeffs0=coeffs0)
        vs, H =  return_dict['grad'], return_dict['H']
        #vs = np.linalg.solve(H + np.eye(d), vs) # experimental to avoid oscillations
        vs = vs[np.newaxis, :]
    else:
        vs, H = compute_gradients(us.T)
    gradterm[it,:,:] = vs.T
    contrterm[it,:,:] = us-m
    #tau = max(tau/100, min(tau, tau*np.max(np.linalg.norm(vs, axis=-1))))
    
    
    if exponential_correction:
        temp1 = us -tau*kappa*vs.T 
        us = m + (temp1-m)*exp(-lam*tau)
        # temp1 = m + (us-m)*exp(-lam*tau)
        # us = temp1-tau*kappa*vs.T 
    else:  
        us = us -tau*kappa*vs.T - lam*tau*(us-m)
    #drift = -tau*kappa*vs.T - lam*tau*(us-m)
    if component_noise:
        diff = sig*(us-m)*np.random.normal(0,sqrt(tau),(d,J))    
    else:
        diff = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
    # project onto orthogonal of drift
    # diff_proj = np.zeros((d,J))
    # for j in range(J):
    #     diff_proj[:,j] = diff[:,j] - (diff[:,j]@drift[:,j])/(drift[:,j]@drift[:,j])*drift[:,j]
    us_new = us + diff#diff_proj
    
    return us_new

# one step of vanilla CBO
def step(us, m):
    drift = - lam*tau*(us-m)
    if component_noise:
        noise = sig*(us-m)*np.random.normal(0,sqrt(tau),(d,J))    
    else:
        noise = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J))    
        
    if exponential_correction:
        return m + (us-m)*exp(-lam*tau)  + noise
    else:
        return us +drift/(1+0.001*np.linalg.norm(drift))  + noise


def step_truegrad(us, m, onlymean=True, tau=1.0, it=0 ):
    gradterm[it,:,:] = DPhi(m)
    contrterm[it,:,:] = us-m
    if exponential_correction:
        print("warning, not yet implemented")
    if onlymean:
        drift = -tau*kappa*DPhi(m)- lam*tau*(us-m)
        return us +drift  + sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J))  
    else:
        drift = -tau*kappa*DPhi(us)- lam*tau*(us-wmean)
        return us +drift  + sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J))  
    
  
#%% prepare animation  

global gradterm
gradterm = np.zeros((N,d,J))
global contrterm 
contrterm = np.zeros((N,d,J))
if MC_test:
    t1 = time.time()
    plt.figure()
    optimizer = np.zeros((N_MC,d))
    for n_MC in range(N_MC):
        print(f"iteration {n_MC}")
        # initial ensemble
        # initial ensemble
        us0 = us0_global
        
        # list of ensemble particles for each iteration
        us_list = np.zeros(( N, d, J))
        us_list[0, :, :] = us0
        
        
        
        w_mean = np.zeros((N,d))
        w_mean[0, :] = weighted_mean(us0.T, lambda u: -alpha*Phi(u))
        
        
        global coeffs_previous 
        coeffs_previous = None
        for i in range(N-1):
            #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
            m = weighted_mean(us_list[i,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
            w_mean[i,:] = m.flatten()
            if i > 0:
                old_pos = (us_list[i-1,:])[np.newaxis, :]
            if use_grad:
                if use_truegrad:
                    us_list[i+1, :] = step_truegrad(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i])
                else:
                    old_pos = None
                    us_list[i+1, :] = step_gradHess(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i], coeffs0=None, memory = old_pos)
            else:    
                us_list[i+1, :] = step(us_list[i, :, :],m)
        m = weighted_mean(us_list[N-1,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
        w_mean[N-1,:] = m.flatten()
        optimizer[n_MC, :] = w_mean[-1, :]
        plt.plot(np.linalg.norm(w_mean, axis=1))
    
    plt.figure()
    plt.hist(Phi(optimizer.T), bins=10)
    
    plt.figure()
    plt.hist(np.linalg.norm(optimizer, axis=1))
    
    sqnorm = 1/d*np.linalg.norm(optimizer, axis=1)**2
    abw = np.max(np.abs(optimizer[:,:]), axis=1)
    print(f"success: {np.sum(abw < 0.25)/N_MC}")
    t2=time.time()
    print(f"elapsed time = {t2-t1}")     

else:    
    from time import time 
    t1 = time()
    for i in range(N-1):
        #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
        m = weighted_mean(us_list[i,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
        w_mean[i,:] = m.flatten()
        if use_grad:
            if use_truegrad:
                us_list[i+1, :] = step_truegrad(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i], it=i)
            else:
                us_list[i+1, :] = step_gradHess(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i], it=i)
        else:    
            us_list[i+1, :] = step(us_list[i, :, :],m)
    print(f"time elapsed: {time()-t1}")
    m = weighted_mean(us_list[N-1,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
    w_mean[N-1,:] = m.flatten()
        

    it = np.linspace(0.0, T, N)
    
    ns = np.arange(N)
    #ts = np.array([it[n] for n in ns])
    # plt.semilogy(it, [Phi(w_mean[n, :]) for n in range(N)], label="val of weighted mean")
    vals = np.log(np.array([Phi(w_mean[n, :]) for n in ns]))
    Phi_mat = np.stack((ts, np.ones_like(ns))).T
    coeff = np.linalg.lstsq(Phi_mat, vals, rcond=None)[0] #np.linalg.solve(Phi_mat.T@Phi_mat, Phi_mat.T@vals)
    
  

    print(w_mean[-1])
    plt.figure(figsize=(5,8))
    plt.subplot(3,1,1)
    plt.semilogy(ts, [Phi(w_mean[n, :]) for n in range(N)])
    plt.title("Phis")
    plt.subplot(3,1,2)
    plt.semilogy(ts, np.sum(np.linalg.norm(us_list-w_mean[:,:,np.newaxis], axis=1), axis=1))
    plt.title("ensemble collapse")
    plt.tight_layout()    
    plt.subplot(3,1,3)
    plt.plot(ts, us_list[:,0,:])
    plt.tight_layout()
    # plt.figure(figsize=(7.5,4))
    # plt.semilogy(ts, Phis_l1d5, label="lamda=1.5")
    # plt.semilogy(ts, Phis_l2d5, label="lamda=2.5")
    # plt.semilogy(ts, Phis_l3d5, label="lamda=3.5")
    # plt.semilogy(ts, Phis_l4d5, label="lamda=4.5")
    # plt.legend()
    # plt.tight_layout()
    
    
    # plt.legend(loc="upper right")
    # for j in range(J):
    #     plt.semilogy(ts, [Phi(us_list[n, :, j]) for n in range(N)], label="val of particle mean")
        
    #plt.semilogy(ts, np.exp(-(lam+kappa-d*sig**2/2)*ts))
    # plt.tight_layout()
    
    # plt.figure()
    # plt.semilogy(ts, [Phi(us_list[i,:,:]) for i in ns], 'r')
    # patch_Phi = mpatches.Patch(color="red", label='Phis')
    # plt.semilogy(ts, np.linalg.norm(gradterm[:,:,:],axis=1), 'k')
    # patch_grad = mpatches.Patch(color="black", label='grad')
    # plt.semilogy(ts, np.exp(-kappa*ts), 'k--')
    # plt.semilogy(ts, np.linalg.norm(contrterm[:,:,:],axis=1), 'b')
    # patch_contr = mpatches.Patch(color="blue", label='contr')
    # handles=[patch_Phi,patch_grad,patch_contr]
    # plt.legend(handles=[h for h in handles if h is not None])
    
    
    # plt.figure()
    # plt.semilogy(ts, np.linalg.norm(gradterm[:,:,0]-us_list[:,:,0], axis=1))
    # plt.title("norm of error in gradient approximation")
    
    # plt.subplot(3,1,2)
    # plt.semilogy(ts, np.sum(np.linalg.norm(us_list-w_mean[:,:,np.newaxis], axis=1), axis=1))
    # plt.title("ensemble collapse")
    # plt.tight_layout()
    
    # plt.figure()
    # plt.semilogy(ts, np.mean(np.linalg.norm(us_list-w_mean[:,:,np.newaxis], axis=2), axis=1), label="deviation")
    # plt.semilogy(ts, np.mean(np.linalg.norm(us_list-0, axis=2), axis=1), label="distance")
    # plt.legend()
    
    
    # angles = np.zeros((N))
    # for n in range(1,N):
    #     g = gradterm[n,:,0]
    #     wm = w_mean[n,:]
    #     angles[n] = g@wm/sqrt(g@g*wm@wm)
        
    # plt.figure()
    # plt.plot(ts[1:], angles[1:])
    
    
    # plt.figure()
    # from math import sqrt
    # nrow = int(d/sqrt(d)) + ceil(d % sqrt(d))
    # ncol = int(d/sqrt(d))
    # for dd in range(d):
    #     plt.subplot(nrow,ncol,dd+1)
    #     plt.plot(ts, us_list[:,dd,:])

    #plt.semilogy(ts, np.linalg.norm(us_list-w_mean[:,:,np.newaxis],axis=1),'g')
    
    # # plot of 1st coordinate
    # # plt.figure(figsize=(5,2))
    # plt.subplot(3,1,3)
    # plt.plot(ts, us_list[:,0,:])
    # plt.tight_layout()
    
    # # plot of ensemble deviations
    # plt.figure()
    # plt.semilogy(ts, np.linalg.norm(us_list-w_mean[:,:,np.newaxis],axis=1))
    # plt.title("distance from weighted mean")
    
# # 2d animation

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(ts, us_list[:,0,:])
# plt.subplot(2,1,2)
# plt.plot(ts, us_list[:,1,:])


# plt.figure()
# plt.plot(us_list[:,0,:], us_list[:,1,:], '.-')
# plt.plot(us_list[0,0,:], us_list[0,1,:], 'r.')
# plt.plot(us_list[int(20//tau),0,:], us_list[int(20//tau),1,:], 'gx')

# fig, ax1 = plt.subplots(1,1, figsize=(5,5))
# xdata, ydata = [], []
# ln1, = ax1.plot([], [], 'ko')
# ln2, = ax1.plot([], [], 'rs')
# ln3, = ax1.plot([],[],'b-')
# xmin = -0.1
# xmax = 0.1
# ymin = -0.1
# ymax = 0.1

# def init():
#     ax1.set_xlim(xmin, xmax)
#     ax1.set_ylim(ymin, ymax)
#     #ax1.contourf(XX, YY, ZZ, 30)
#     return ln1, ln2,

# def update(frame_num): # expects frame to contain the new position of the ensemble
#     xdata1 = us_list[frame_num, 0, :]
#     ydata1 = us_list[frame_num, 1, :]
#     ln1.set_data(xdata1, ydata1)
#     ln2.set_data([w_mean[frame_num, 0], 0], [w_mean[frame_num, 1], 0])
    
#     ax1.set_title(f"t={frame_num*T/N:.2f}")
#     mx = np.mean(xdata1)
#     my = np.mean(ydata1)
#     dx = max(xdata1)-min(xdata1)
#     dy = max(ydata1)-min(ydata1)
#     dd = max(dx, dy)
#     center = w_mean[frame_num, :]
#     factor = sqrt(center@center)
#     ln3.set_data([w_mean[frame_num, 0], (1-dd)/factor*w_mean[frame_num, 0]],[w_mean[frame_num, 1], (1-dd)/factor*w_mean[frame_num, 1]])
#     ax1.set_xlim((mx-dd,mx+dd))
#     ax1.set_ylim((my-dd,my+dd))
#     return ln1, 

    
# ani = FuncAnimation(fig, update, frames = range(0,N,2), init_func=init, blit=False, interval=2)   

# # plt.figure()
# # plt.plot(us_list[nn,0,:], us_list[nn,1,:], '.')
# # plt.quiver(us_list[nn,0,:], us_list[nn,1,:], -us_list[nn,0,:], -us_list[nn,1,:])
# # plt.plot(0,0,'rx')
"""