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

from grad_inference import *
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(1)


# Himmelblau
fnc = lambda x, y: (x**2+y-11)**2 + (x+y**2-7)**2
Dfnc = lambda x, y: np.array([2*(x**2+y-11)*2*x + 2*(x+y**2-7), 2*(x**2+y-11) + 2*(x+y**2-7)*2*y])

# Rastrigin
fnc = lambda x, y: 10*2 + x**2 - 10*np.cos(2*pi*x) + y**2 - 10*np.cos(2*pi*y)
Dfnc = lambda x, y: np.array([2*x + 2*pi*10*np.sin(2*pi*x),  2*y + 2*pi*10*np.sin(2*pi*y)])
D2fnc = lambda x,y:np.array([[2 + 4*pi**2*10*np.cos(2*pi*x), 0], [0, 2 + 4*pi**2*10*np.cos(2*pi*y)]])

# Ackley
#fnc = lambda x, y: -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*np.cos(2*pi*x)+0.5*np.cos(2*pi*y)) + e + 20



Phi = lambda z: fnc(z[0], z[1])# just for safety
DPhi = lambda z: Dfnc(z[0], z[1])


# square_exact = lambda x: fnc(wmean[0],wmean[1]) + Dfnc(wmean[0],wmean[1]).flatten()@(x-wmean) + 0.5*(x-wmean)@(D2fnc(wmean[0][0],wmean[1][0])@(x-wmean))

# make full Monte Carlo simulation?
MC_test = False
N_MC = 20

# global switch: gradient-based yes or no?
use_grad = True
onlymean = True # compute gradient only in mean?

use_truegrad = False # use true gradient instead of inferred Bayesian gradient?
component_noise = True
orthogonal_noise = False

xmin = -4.5
xmax = 1.5
ymin = -4.5
ymax = 1.5


xs_plot = np.linspace(xmin,xmax,50)
ys_plot = np.linspace(ymin,ymax,50)
XX, YY = np.meshgrid(xs_plot, ys_plot)
ZZ = fnc(XX, YY)

d = 2




T = 20.0 


tau = 0.01
N = int(T/tau)
deltas = [tau for k in range(1,N)]
ts = np.arange(0,T,tau)



# new timing scheme
# r = 0.25
# delta0 = 0.01
# N = 2000
# deltas = [delta0*k**(-r) for k in range(1,N)]
# ts = np.cumsum(deltas)


alpha = 100.0 # weight exponent of cost function exp(-alpha*Phi)
lam = 1.5 # coefficient of contraction towards weighted mean
sig = 0.7 # coefficient of noise
kappa = 0.5 # coefficient of gradient drift term

N_ens = 2 # size of ensemble
J = N_ens
avar = 0.1

# inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False)


# initial ensemble
us0_global = np.random.uniform(-4,-1,(2,J)) # hard for functions with minimum at (0,0)
us0_global = np.array([[-4,-1.5],[-4,-2.5]])
#us0_global = np.random.uniform(-2,2,(2,J)) # hard for Himmelblau


# list of ensemble particles for each iteration
us_list = np.zeros(( N, d, J))
us_list[0, :, :] = us0_global

#%%

# plot initial config

plt.figure(figsize=(5,5))
plt.contourf(XX, YY, ZZ, 30)
plt.plot(us_list[0, 0, :], us_list[0, 1, :], '.k')

w_mean = np.zeros((N,len(us0_global)))
w_mean[0, :] = weighted_mean(us0_global.T, lambda u: -alpha*Phi(u))
plt.plot(w_mean[0, 0], w_mean[0,1], 'rx')



# method for computing all gradients at once
def compute_gradients(points):
    vs = np.zeros_like(points)
    H = np.zeros((points.shape[0],points.shape[1],points.shape[1]))
    for i in range(points.shape[0]):
        return_dict= inferGradientAndHess(points.T, Phi(points.T).flatten(), hessian = True, ind=i, additionalvariance=avar)
        vs[i, :], H[i,:,:] = return_dict['grad'], return_dict['H']
    return vs, H


# one step of gradient-augmented CBO
def step_gradHess(us, wmean, onlymean=True, tau=1.0):
    #m = weighted_mean(us_list.reshape((-1,1)), lambda u: -alpha*Phi(u))
    if onlymean:
        ensembleplusmean = np.concatenate((wmean,us),axis=1)
        values = np.concatenate((Phi(wmean),Phi(us)))
        return_dict= inferGradientAndHess(ensembleplusmean, values, hessian=True, ind=0, additionalvariance=avar)
        vs, H =  return_dict['grad'], return_dict['H']
        #vs = np.linalg.solve(H + np.eye(d), vs) # experimental to avoid oscillations
        vs = vs[np.newaxis, :]
    else:
        vs, H = compute_gradients(us.T)
    if component_noise:
        diff = sig*(us-m)*np.random.normal(0,sqrt(tau),(d,J))    
    else:
        if orthogonal_noise:
            diff0 = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
            us_step = us + 0.5*diff0
            m_step = weighted_mean(us_step.T, lambda u: -alpha*Phi(u)).reshape((-1,1))
            
            term1 = (np.einsum("ij,ij->j", (us_step-m_step), diff0))
            term2 = (np.einsum("ij,ij->j", (us_step-m_step), (us_step-m_step)))
            division = np.divide(term1, term2, where=term2 > 0)
            diff_orth = diff0 - division*(us_step-m_step)
            return wmean + (us-wmean + diff_orth - tau*kappa*vs.T)*exp(-tau*lam)
        else:
            diff = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
    drift = -tau*kappa*vs.T - lam*tau*(us-wmean)
    us_new = us +drift  + diff#sig*np.linalg.norm(us-wmean, axis=0)*np.random.normal(0,sqrt(tau),(d,J))    
    return us_new

# one step of vanilla CBO
def step(us, wmean):
    drift = - lam*tau*(us-wmean)
    if component_noise:
        noise = sig*(us-m)*np.random.normal(0,sqrt(tau),(d,J))    
    else:
        noise = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J))  
        if orthogonal_noise:
            diff0 = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
            us_step = us + 0.5*diff0
            m_step = weighted_mean(us_step.T, lambda u: -alpha*Phi(u)).reshape((-1,1))
            term1 = (np.einsum("ij,ij->j", (us_step-m_step), diff0))
            term2 = (np.einsum("ij,ij->j", (us_step-m_step), (us_step-m_step)))
            division = np.divide(term1, term2, where=term2 > 0)
            diff_orth = diff0 - division*(us_step-m_step)
            return wmean + (us-wmean + diff_orth)*exp(-tau*lam)
        else:
            diff = sig*np.linalg.norm(us-m, axis=0)*np.random.normal(0,sqrt(tau),(d,J)) 
    return us +drift/(1+0.001*np.linalg.norm(drift))  + noise


def step_truegrad(us, wmean, onlymean=True, tau=1.0 ):
    if onlymean:
        drift = -tau*kappa*DPhi(wmean)- lam*tau*(us-wmean)
        return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.linalg.norm(us-wmean, axis=0)*np.random.normal(0,sqrt(tau),(d,J))  
    else:
        drift = -tau*kappa*DPhi(us)- lam*tau*(us-wmean)
        return us +drift/(1+0.001*np.linalg.norm(drift))  + sig*np.linalg.norm(us-wmean, axis=0)*np.random.normal(0,sqrt(tau),(d,J))  
    
  
#%% prepare animation  

if MC_test:
    t1 = time.time()
    optimizer = np.zeros((N_MC,2))
    for n_MC in range(N_MC):
        # initial ensemble
        # initial ensemble
        us0 = us0_global
        
        # list of ensemble particles for each iteration
        us_list = np.zeros(( N, d, J))
        us_list[0, :, :] = us0
        
        
        
        w_mean = np.zeros((N,d))
        w_mean[0, :] = weighted_mean(us0.T, lambda u: -alpha*Phi(u))
        
        
        
        for i in range(N-1):
            #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
            m = weighted_mean(us_list[i,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
            w_mean[i,:] = m.flatten()
            if use_grad:
                if use_truegrad:
                    us_list[i+1, :] = step_truegrad(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i])
                else:
                    us_list[i+1, :] = step_gradHess(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i])
            else:    
                us_list[i+1, :] = step(us_list[i, :, :],m)
        m = weighted_mean(us_list[N-1,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
        w_mean[N-1,:] = m.flatten()
        optimizer[n_MC, :] = w_mean[-1, :]
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
        ax1.contourf(XX, YY, ZZ, 30)
        return ln1, ln2,
    
    def update(frame_num): # expects frame to contain the new position of the ensemble
        xdata1 = us_list[frame_num, 0, :]
        ydata1 = us_list[frame_num, 1, :]
        ln1.set_data(xdata1, ydata1)
        ln2.set_data(w_mean[frame_num, 0], w_mean[frame_num, 1])
        ax1.set_title(f"it={frame_num}")
        return ln1, 
    
    ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=20)
    # new transparency-augmented color map for plotting histogram on top of utility function
    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # Choose colormap
    cmap = pl.cm.plasma    
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))    
    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)    
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    
    
    
    fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.contourf(XX, YY, ZZ, 30)
    ax2.plot(optimizer[:,0], optimizer[:, 1], 'k.')
    H, xedges, yedges =np.histogram2d(optimizer[:,0],optimizer[:,1], range=[[-4.25,4.25],[-4.25,4.25]], bins=17)
    ax2.pcolormesh(xedges, yedges, H.T, cmap=my_cmap)
    print(f"success rate: {np.sum(np.linalg.norm(optimizer, axis=1) <= 1e-2)/N_MC}")

else:    
    fig, ax1 = plt.subplots(1,1, figsize=(5,5))
    xdata, ydata = [], []
    ln1, = ax1.plot([], [], 'ko')
    ln2, = ax1.plot([], [], 'rs')
    #plt.plot(xs, ys)
    #plt.title("target function")
    
    def init():
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.contourf(XX, YY, ZZ, 30)
        return ln1, ln2,
    
    def update(frame_num): # expects frame to contain the new position of the ensemble
        xdata1 = us_list[frame_num, 0, :]
        ydata1 = us_list[frame_num, 1, :]
        ln1.set_data(xdata1, ydata1)
        ln2.set_data(w_mean[frame_num, 0], w_mean[frame_num, 1])
        ax1.set_title(f"t={frame_num*T/N:.2f}")
        return ln1, 
    
    for i in range(N-1):
        #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
        m = weighted_mean(us_list[i,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
        w_mean[i,:] = m.flatten()
        if use_grad:
            if use_truegrad:
                us_list[i+1, :] = step_truegrad(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i])
            else:
                us_list[i+1, :] = step_gradHess(us_list[i, :, :],m,onlymean=onlymean, tau=deltas[i])
        else:    
            us_list[i+1, :] = step(us_list[i, :, :],m)
    m = weighted_mean(us_list[N-1,:, :].T, lambda u: -alpha*Phi(u)).reshape((-1,1))
    w_mean[N-1,:] = m.flatten()
        
    ani = FuncAnimation(fig, update, frames = range(N), init_func=init, blit=False, interval=2)
    
    
    # f = r"P://Philipp/Erlangen/Forschung/Bayesian_CBO/animation.gif" 
    # writergif = PillowWriter(fps=30) 
    # anim.save(f, writer=writergif)


    plt.figure()
    plt.subplot(2,1,1)
    plt.title("weighted mean")
    it = np.linspace(0.0, T, N)
    
    ns = np.arange(N)
    ts = np.array([it[n] for n in ns])
    plt.semilogy(it, [Phi(w_mean[n, :]) for n in range(N)], label="val of weighted mean")
    vals = np.log(np.array([Phi(w_mean[n, :]) for n in ns]))
    Phi_mat = np.stack((ts, np.ones_like(ns))).T
    coeff = np.linalg.lstsq(Phi_mat, vals, rcond=None)[0] #np.linalg.solve(Phi_mat.T@Phi_mat, Phi_mat.T@vals)
    
  
    plt.semilogy(ts, np.exp(coeff[1] + coeff[0]*ts), label=f"$\exp({coeff[1]:.2f} {coeff[0]:.2f}\cdot t)$")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.loglog(it, [Phi(w_mean[n, :]) for n in range(N)], label="val of weighted mean")
    plt.plot(it, np.exp(coeff[1] + coeff[0]*ts), label=f"$\exp({coeff[1]:.2f} {coeff[0]:.2f}\cdot t)$")
    plt.legend()
    
    
    plt.figure(figsize=(5,4))
    plt.subplot(2,1,1)
    plt.title("weighted mean")
    # it = np.linspace(0.0, T, N)
    
    
    ns = np.arange(N)
    #ts = np.array([it[n] for n in ns])
    plt.semilogy(ts, [Phi(w_mean[n, :]) for n in range(N)], label="val of weighted mean")
    plt.subplot(2,1,2)
    for p in range(J):
        plt.semilogy(ts, Phi(us_list[:,:,p].T))
    plt.tight_layout()
    
    
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ts, us_list[:,0,:])
    plt.subplot(2,1,2)
    plt.plot(ts, us_list[:,1,:])
    
    plt.figure()
    plt.semilogy(np.linalg.norm(us_list-w_mean[:,:,np.newaxis],axis=1))
    plt.semilogy(np.mean(np.linalg.norm(us_list-w_mean[:,:,np.newaxis],axis=1),axis=-1), 'k', linewidth=3)
    plt.title("collapse")
    
    
    # code for plot of gradients
    
    plt.figure()
    plt.contourf(XX, YY, ZZ, 30)
    xs = us_list[-1, :, :]
    vs = Phi(xs)
    plt.plot(xs[0,:], xs[1,:], '.')
    grads = []
    HS = []
    if onlymean:
        m = weighted_mean(xs.T, lambda u: -alpha*Phi(u)).reshape((-1,1))
        wmean = m.flatten()
        ensembleplusmean = np.concatenate((wmean[:,np.newaxis],
                                           xs),axis=1)
        values = np.concatenate(([Phi(wmean)],vs))
        return_dict= inferGradientAndHess(ensembleplusmean, values, hessian=True, ind=0)
        grads, H =  return_dict['grad'], return_dict['H']
    for k in range(J):
        if onlymean:
            plt.quiver(xs[0,k], xs[1,k], grads[0], grads[1], color="blue", label="max likelihood")
        else:
            ret = inferGradientAndHess(xs, vs, hessian=True,ind=k,penalizedistance=True)
            gk, Hk = ret['grad'], ret['H']
            grads = [*grads, gk]
            HS = [*HS, Hk]
            plt.quiver(xs[0,k], xs[1,k], grads[k][0], grads[k][1], color="blue", label="max likelihood")
