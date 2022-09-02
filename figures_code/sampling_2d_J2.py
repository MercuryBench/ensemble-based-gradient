# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:40:42 2022

@author: Philipp
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, pi, exp
import scipy.linalg
from grad_inference import *
import time

np.random.seed(1)

alpha = 1.0
J = 2
N_sim = 10000

#-----------------------------------
# test case "banana"
y = 0
sigNoise = 0.5
sigPrior = 2
tau = 0.01
u0 = np.random.normal(0,1.0,(2,J))

xmin = -2
xmax = 7
ymin= -1
ymax= 5



G = lambda u: ((u[1]-2)**2-(u[0]-3.5) -1)
Phi = lambda u: 0.5/(sigNoise**2)*(G(u)-y)**2


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
# y = np.array([27.5,79.7])
# sigNoise = 0.4
# sigPrior = 10
# tau = 0.005
# u0 = np.random.normal(0,0.5,(2,J)) + np.array([[-3.0],[103]])

# xmin = -3.75
# xmax = -1.75
# ymin =  100
# ymax =  106

# xobs = np.array([0.25, 0.75])

# p = lambda x, u: np.tensordot(u[1],x,axes=0) + np.tensordot(np.exp(-u[0]),(-x**2/2+x/2), axes=0)
# G = lambda u: p(xobs, u)
# Phi = lambda u: 0.5/(sigNoise**2)*np.linalg.norm(G(u)-y, axis=-1)**2



# parameters independent of test case



I = lambda u: Phi(u) + 0.5/(sigPrior**2)*np.linalg.norm(u,axis=0)**2

#%%
### Test of specification
# print(p(xobs,u0))
# print(Phi(u0))


u0s = np.linspace(xmin,xmax,150)
u1s = np.linspace(ymin,ymax,150)
U0, U1 = np.meshgrid(u0s,u1s)
U = np.stack((U0,U1))


# print(p(xobs, U).shape)
# print(Phi(U).shape)

plt.figure()
plt.contourf(U0, U1, np.exp(-I(U)), 10)
plt.title("posterior")

unnorm_dens = np.exp(-I(U))
Z = np.trapz(unnorm_dens, u0s, axis=1)
Z = np.trapz(Z, u1s)
dens = unnorm_dens/Z

marg_over_x = np.trapz(dens, u0s, axis=1)
marg_over_y = np.trapz(dens, u1s, axis=0)

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(u0s, marg_over_y)
# plt.subplot(2,1,2)
# plt.plot(u1s, marg_over_x)

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
    

# plt.quiver(u0[0], u0[1], Dfnc(u0)[0], Dfnc(u0)[1])
# vs, H = compute_gradients(u0)
# plt.quiver(u0[0], u0[1], vs[0], vs[1], color="blue")

# u1 =  u0- tau*vs + sqrt(2/alpha)*np.random.normal(0,sqrt(tau),(2,J))

# connections = np.stack((u0,u1),axis=1)

# plt.plot(connections[0], connections[1],'-')
# plt.scatter(u0[0], u0[1], color="blue")

# plt.scatter(u1[0], u1[1], color="red")

t1 = time.time()
# Langevin
us_list = np.zeros((2,J,N_sim))
us_list[:,:,0] = u0


for n in range(N_sim-1):    
    us = us_list[:,:,n]
    vs, H = compute_gradients(us_list[:,:,n])
    #vs = DI(us_list[:,:,n])
    tauk = tau * np.linalg.norm(us, axis=0)/(1e-3+np.linalg.norm(vs, axis=0))
    us_list[:,:,n+1] = us_list[:,:,n] - tauk*alpha*vs + sqrt(2)*tauk**(1/2)*np.random.normal(0,1,(2,J))

print(f"Langevin: {time.time()-t1} seconds")

#%%
# MALA
t2 = time.time()
us_list_MALA = np.zeros((2,J,N_sim))
us_list_MALA[:,:,0] = u0
total_acc = 0
running_memory = np.zeros((2,J))
for n in range(N_sim-1):    
    us = us_list_MALA[:,:,n]    
    if n >= 1: 
        vs, H = compute_gradients_withmemory(us_list_MALA[:,:,n], running_memory)
    else:
        vs, H = compute_gradients(us_list_MALA[:,:,n])
    #vs = DI(us_list_MALA[:,:,n])
    
    tauk = tau#tau * np.linalg.norm(us, axis=0)/(1e-3 + np.linalg.norm(vs, axis=0))
    
    props = us_list_MALA[:,:,n] - tauk*alpha*vs + sqrt(2)*tauk**(1/2)*np.random.normal(0,1,(2,J))
    vs_props, _ = compute_gradients(props)
    if n >= 1: 
        vs_props, _ = compute_gradients_withmemory(props, running_memory)
    else:
        vs_props, _ = compute_gradients(props)
    #vs_props = vs
    
    #vs_props = DI(props)
    propkernel1 = np.exp(-1/(4*tauk)*np.linalg.norm(us_list_MALA[:,:,n] - (props - tauk*alpha*vs_props), axis=0)**2)
    propkernel2 = np.exp(-1/(4*tauk)*np.linalg.norm(props - (us_list_MALA[:,:,n] - tauk*alpha*vs), axis=0)**2)
    acc = np.exp(alpha*(I(us_list_MALA[:,:,n])-I(props))) * propkernel1/propkernel2
    rnds = np.random.uniform(0,1,J)
    us_list_MALA[:,:,n+1] = np.where(rnds <= acc, props, us_list_MALA[:,:,n])
    running_memory = np.where(rnds > acc, props, us_list_MALA[:,:,n]) # take complement for better gradient approximation
    total_acc += np.sum(rnds <= acc)

total_acc /= J*N_sim
print(f"acceptance ratio: {total_acc}")
print(f"MALA: {time.time()-t2} seconds")

#%%
# Consensus-based sampling
t2a = time.time()
us_list_CBS = np.zeros((2,J,N_sim))
us_list_CBS[:,:,0] = u0
tau_CBS = 10*tau

alpha = exp(-tau_CBS)

lam = (1+1)**(-1)

for n in range(N_sim-1):    
    us = us_list_CBS[:,:,n]    
    
    # compute weights
    logweights = -I(us)
    logsumweights = logsumexp(logweights)
    logweights = logweights - logsumweights
    weights = np.exp(logweights)
    
    # compute weighted mean
    wmean = us @ weights
    wmean_resized = wmean[:,np.newaxis]
    
    wcov = np.einsum("ij,kj,j->ik", us-wmean_resized, us-wmean_resized, weights)
    wcov_sqrt = scipy.linalg.sqrtm(wcov)#np.einsum("ij,j->i", us-wmean_resized, np.sqrt(weights))[:,np.newaxis]
    
    us_list_CBS[:,:,n+1] = wmean_resized + alpha*(us-wmean_resized) + sqrt((1-alpha**2)*(lam)**(-1))*wcov_sqrt@np.random.normal(0,1,(2,J))
    
    #us_list_CBS[:,:,n+1] = us - tau*(us-wmean_resized) + sqrt(4*tau)*wcov@np.random.normal(0,1,(2,12))
print(f"CBS: {time.time()-t2a} seconds")


#%%

# ALDI
t3 = time.time()
us_list_ALDI = np.zeros((2,J,N_sim))
us_list_ALDI[:,:,0] = u0
total_acc = 0
tau_ALDI = tau

if hasattr(y, "__len__"):
    y_algo = y[:,np.newaxis]
else:
    y_algo = np.array([[y]])
for n in range(N_sim-1):   
    us = us_list_ALDI[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    if G_us_unprocessed.ndim == 1: # this is just to catch an annoying thing when G has higher dimension
        G_us = G_us_unprocessed[np.newaxis,:]
        m_G_us = np.mean(G_us, axis=1)[np.newaxis, :]
    else:
        G_us = G_us_unprocessed.T
        m_G_us = np.mean(G_us, axis=1)[:,np.newaxis]
    u_c = us-m_us
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c)
    C = np.cov(us)*(J-1)/J#np.mean((us-m_us)**2,axis=1)
    E = np.cov(G_us)*(J-1)/J
    Csqrt = 1/sqrt(J)*u_c
    
    drift = -1/(sigNoise**2 + tau_ALDI*E)*D@(G_us-y_algo) - 1/sigPrior**2*C@us + 3*1/J*(us-m_us)
    diff = sqrt(2)*Csqrt@np.random.normal(0,1,(J,J))
    
    #tauk = tau * np.linalg.norm(us, axis=0)/np.linalg.norm(drift, axis=0)
    
    us_list_ALDI[:,:,n+1] = us+tau_ALDI*drift  + sqrt(tau_ALDI)*diff
print(f"ALDI: {time.time()-t3} seconds")


#%%


# augmented ALDI
t5 = time.time()
us_list_augALDI = np.zeros((2,J,N_sim))
us_list_augALDI[:,:,0] = u0
total_acc = 0
tau_ALDI = tau

if hasattr(y, "__len__"):
    y_algo = y[:,np.newaxis]
else:
    y_algo = np.array([[y]])
for n in range(N_sim-1):   
    us = us_list_augALDI[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    if G_us_unprocessed.ndim == 1: # this is just to catch an annoying thing when G has higher dimension
        G_us = G_us_unprocessed[np.newaxis,:]
        m_G_us = np.mean(G_us, axis=1)[np.newaxis, :]
    else:
        G_us = G_us_unprocessed.T
        m_G_us = np.mean(G_us, axis=1)[:,np.newaxis]
    u_c = us-m_us
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c)
    C = np.cov(us)*(J-1)/J#np.mean((us-m_us)**2,axis=1)
    E = np.cov(G_us)*(J-1)/J
    Csqrt = 1/sqrt(J)*u_c
    
    vs, H = compute_gradients(us_list_augALDI[:,:,n])
    
    #drift = -(sigNoise**2 + tau_ALDI*E)**(-1)*C@vs- 1/sigPrior**2*C@us + 3*1/J*(us-m_us)
    #drift = -C@vs- 1/sigPrior**2*C@us + 3*1/J*(us-m_us)
    drift = -C@vs + 3*1/J*(us-m_us)
    diff = sqrt(2)*Csqrt@np.random.normal(0,1,(J,J))
    
    #tauk = tau * np.linalg.norm(us, axis=0)/np.linalg.norm(drift, axis=0)
    
    us_list_augALDI[:,:,n+1] = us+tau_ALDI*drift  + sqrt(tau_ALDI)*diff
print(f"augmented ALDI: {time.time()-t5} seconds")


#%%

# augmented ALDI with one gradient
t6 = time.time()
us_list_augALDIone = np.zeros((2,J,N_sim))
us_list_augALDIone[:,:,0] = u0
total_acc = 0
tau_ALDI = tau

if hasattr(y, "__len__"):
    y_algo = y[:,np.newaxis]
else:
    y_algo = np.array([[y]])
for n in range(N_sim-1):   
    us = us_list_augALDIone[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    if G_us_unprocessed.ndim == 1: # this is just to catch an annoying thing when G has higher dimension
        G_us = G_us_unprocessed[np.newaxis,:]
        m_G_us = np.mean(G_us, axis=1)[np.newaxis, :]
    else:
        G_us = G_us_unprocessed.T
        m_G_us = np.mean(G_us, axis=1)[:,np.newaxis]
    u_c = us-m_us
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c)
    C = np.cov(us)*(J-1)/J#np.mean((us-m_us)**2,axis=1)
    E = np.cov(G_us)*(J-1)/J
    Csqrt = 1/sqrt(J)*u_c
    
    
    points = np.concatenate((m_us, us), axis=1)
    
    return_dict= inferGradientAndHess(points, I(points), hessian = True, ind=0, additionalvariance=0.0)
    v, H = return_dict['grad'], return_dict['H']
    vs = v[:,np.newaxis] + H@(us-m_us)
    
    #drift = -(sigNoise**2 + tau_ALDI*E)**(-1)*C@vs- 1/sigPrior**2*C@us + 3*1/J*(us-m_us)
    #drift = -C@vs- 1/sigPrior**2*C@us + 3*1/J*(us-m_us)
    drift = -C@vs + 3*1/J*(us-m_us)
    diff = sqrt(2)*Csqrt@np.random.normal(0,1,(J,J))
    
    #tauk = tau * np.linalg.norm(us, axis=0)/np.linalg.norm(drift, axis=0)
    
    us_list_augALDIone[:,:,n+1] = us+tau_ALDI*drift  + sqrt(tau_ALDI)*diff
print(f"augmented ALDI (one): {time.time()-t6} seconds")

#%%
########## RW-MH als Goldstandard

t4 = time.time()
us_list_RWMH = np.zeros((2,J,N_sim))
us_list_RWMH[:,:,0] = u0
total_acc = 0

propgamma = 30*tau
for n in range(N_sim-1):    
    us = us_list_RWMH[:,:,n]    
    
    props = us + np.random.normal(0,propgamma,(2,J))
    # plt.figure()
    # plt.scatter(us[0,:],us[1,:], color="black")
    # plt.scatter(props[0,:],props[1,:], color="red")
    # for j in range(J):
    #     plt.plot([us[0,j], props[0,j]], [us[1,j],props[1,j]], '--k')
    acc = np.exp(alpha*(I(us)-I(props))) 
    rnds = np.random.uniform(0,1,J)
    for j in range(J):
        if rnds[j] <= acc[j]:
            us_list_RWMH[:,j,n+1] = props[:,j]
        else:
            us_list_RWMH[:,j,n+1] = us_list_RWMH[:,j,n]
    #us_list_RWMH[:,:,n+1] = np.where(rnds <= acc, props, us)
    #running_memory = np.where(rnds > acc, props, us_list_RWMH[:,:,n]) # take complement for better gradient approximation
    total_acc += np.sum(rnds <= acc)

total_acc /= J*N_sim
print(f"acceptance ratio RWMH: {total_acc}")

#%%

import matplotlib.pylab as pl

N_burnin = 0#1000#int(N_sim/2)

binsx = np.linspace(xmin,xmax,31)
binsy = np.linspace(ymin,ymax,31)
H, yedges, xedges = np.histogram2d(us_list[0,:,N_burnin:].flatten(),us_list[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H2, yedges2, xedges2 = np.histogram2d(us_list_MALA[0,:,N_burnin:].flatten(),us_list_MALA[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H3, yedges3, xedges3 = np.histogram2d(us_list_ALDI[0,:,N_burnin:].flatten(),us_list_ALDI[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H4, yedges4, xedges4 = np.histogram2d(us_list_RWMH[0,:,N_burnin:].flatten(),us_list_RWMH[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H5, yedges5, xedges5 = np.histogram2d(us_list_augALDI[0,:,N_burnin:].flatten(),us_list_augALDI[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H6, yedges6, xedges6 = np.histogram2d(us_list_CBS[0,:,N_burnin:].flatten(),us_list_CBS[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
H7, yedges7, xedges7 = np.histogram2d(us_list_augALDIone[0,:,N_burnin:].flatten(),us_list_augALDIone[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))= plt.subplots(ncols=3, nrows=2,figsize=(10,15))



ax1.pcolormesh(yedges, xedges, H.T, cmap=pl.cm.viridis_r); 
ax1.set_xlim((xmin,xmax));ax1.set_ylim((ymin,ymax))
ax1.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax1.set_title("ELSa")

ax2.pcolormesh(yedges2, xedges2, H2.T, cmap=pl.cm.viridis_r); 
ax2.set_xlim((xmin,xmax));ax2.set_ylim((ymin,ymax))
ax2.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax2.set_title("EMALA")

ax4.pcolormesh(yedges3, xedges3, H3.T, cmap=pl.cm.viridis_r); 
ax4.set_xlim((xmin,xmax));ax3.set_ylim((ymin,ymax))
ax4.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax4.set_title("ALDI")

ax6.pcolormesh(yedges7, xedges7, H7.T, cmap=pl.cm.viridis_r); 
#ax6.pcolormesh(yedges4, xedges4, H4.T, cmap=pl.cm.viridis_r); 
ax6.set_xlim((xmin,xmax));ax4.set_ylim((ymin,ymax))
ax6.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax6.set_title("ALDI-extra")#ax6.set_title("RWMH")

ax5.pcolormesh(yedges5, xedges5, H5.T, cmap=pl.cm.viridis_r);  
ax5.set_xlim((xmin,xmax));ax5.set_ylim((ymin,ymax))
ax5.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax5.set_title("ALDI-est")

ax3.pcolormesh(yedges6, xedges6, H6.T, cmap=pl.cm.viridis_r); 
ax3.set_xlim((xmin,xmax));ax6.set_ylim((ymin,ymax))
ax3.contour(U0, U1, np.exp(-I(U)), 5, alpha=0.4, colors="black")
ax3.set_title("CBS")


plt.figure()
plt.contourf(U0, U1, I(U), 10)
randint = np.random.randint(0,J)
plt.plot(us_list_MALA[0,randint,N_burnin:],us_list_MALA[1,randint,N_burnin:], 'b.-', label="MALA")
plt.plot(us_list[0,randint,N_burnin:],us_list[1,randint,N_burnin:], 'r.-', label="Langevin")
plt.plot(us_list_ALDI[0,randint,N_burnin:],us_list_ALDI[1,randint,N_burnin:], 'g.-', label="ALDI")
plt.plot(us_list_augALDI[0,randint,N_burnin:],us_list_augALDI[1,randint,N_burnin:], 'm.-', label="aug ALDI")
plt.plot(us_list_CBS[0,randint,N_burnin:],us_list_CBS[1,randint,N_burnin:], 'm.-', label="CBS")
plt.plot(us_list_RWMH[0,randint,N_burnin:],us_list_RWMH[1,randint,N_burnin:], 'k.-', label="RWMH")
plt.legend()

us_list_MALA_burnin = us_list_MALA[:,:,N_burnin:]
us_list_burnin = us_list[:,:,N_burnin:]
us_list_ALDI_burnin = us_list_ALDI[:,:,N_burnin:]
us_list_augALDI_burnin = us_list_augALDI[:,:,N_burnin:]
us_list_augALDIone_burnin = us_list_augALDIone[:,:,N_burnin:]
us_list_RWMH_burnin = us_list_RWMH[:,:,N_burnin:]
us_list_CBS_burnin = us_list_CBS[:,:,N_burnin:]


samples = us_list_burnin.reshape((2,-1))
print(f"Langevin: mean={np.mean(samples,axis=1)}, cov={np.cov(samples)}")
samples_MALA = us_list_MALA_burnin.reshape((2,-1))
print(f"MALA: mean={np.mean(samples_MALA,axis=1)}, cov={np.cov(samples_MALA)}")
samples_ALDI =us_list_ALDI_burnin.reshape((2,-1))
print(f"ALDI: mean={np.mean(samples_ALDI,axis=1)}, cov={np.cov(samples_ALDI)}")
samples_augALDI =us_list_augALDI_burnin.reshape((2,-1))
print(f"augmented ALDI: mean={np.mean(samples_augALDI,axis=1)}, cov={np.cov(samples_augALDI)}")
samples_augALDIone =us_list_augALDIone_burnin.reshape((2,-1))
print(f"augmented ALDI: mean={np.mean(samples_augALDI,axis=1)}, cov={np.cov(samples_augALDI)}")
samples_RWMH =us_list_RWMH_burnin.reshape((2,-1))
print(f"RWMH: mean={np.mean(samples_RWMH,axis=1)}, cov={np.cov(samples_RWMH)}")
samples_CBS =us_list_CBS_burnin.reshape((2,-1))
print(f"CBS: mean={np.mean(samples_CBS,axis=1)}, cov={np.cov(samples_CBS)}")

alpha_op = 1.0
plt.figure()

plt.subplot(3,2,1)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_burnin[0,i,:],us_list_burnin[1,i,:], 'k.', alpha=alpha_op, label="Langevin")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("Langevin")
plt.subplot(3,2,2)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_MALA_burnin[0,i,:],us_list_MALA_burnin[1,i,:], 'k.', alpha=alpha_op, label="MALA")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("MALA")
plt.subplot(3,2,3)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_ALDI_burnin[0,i,:],us_list_ALDI_burnin[1,i,:], 'k.', alpha=alpha_op, label="ALDI")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("ALDI")
plt.subplot(3,2,4)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_RWMH_burnin[0,i,:],us_list_RWMH_burnin[1,i,:], 'k.', alpha=alpha_op, label="RWMH")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("RWMH")
plt.subplot(3,2,5)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_augALDI_burnin[0,i,:],us_list_augALDI_burnin[1,i,:], 'k.', alpha=alpha_op, label="ALDI")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("ALDI-est")
plt.subplot(3,2,6)
plt.contourf(U0, U1, I(U), 10)
for i in range(J):
    plt.plot(us_list_CBS_burnin[0,i,:],us_list_CBS_burnin[1,i,:], 'k.', alpha=alpha_op, label="ALDI")
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.title("CBS")

plt.figure()
plt.subplot(2,3,1)
plt.title("ELSa")
for i in range(J):
    plt.plot(us_list_burnin[0,i,:], alpha=alpha_op)
plt.subplot(2,3,2)
plt.title("EMALA")
for i in range(J):
    plt.plot(us_list_MALA_burnin[0,i,:], alpha=alpha_op)
plt.subplot(2,3,4)
plt.title("ALDI")
for i in range(J):
    plt.plot(us_list_ALDI_burnin[0,i,:], alpha=alpha_op)
plt.subplot(2,3,6)
plt.title("RWMH")
for i in range(J):
    plt.plot(us_list_RWMH_burnin[0,i,:], alpha=alpha_op)
plt.subplot(2,3,5)
plt.title("ALDI-est")
for i in range(J):
    plt.plot(us_list_augALDI_burnin[0,i,:], alpha=alpha_op)
plt.subplot(2,3,3)
plt.title("CBS")
for i in range(J):
    plt.plot(us_list_CBS_burnin[0,i,:], alpha=alpha_op)
plt.tight_layout()

plt.figure()
plt.subplot(2,3,1)
plt.title("ELSa")
plt.hist(samples[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.subplot(2,3,2)
plt.title("EMALA")
plt.hist(samples_MALA[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.subplot(2,3,4)
plt.title("ALDI")
plt.hist(samples_ALDI[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.subplot(2,3,6)
#plt.title("RWMH")
plt.title("ALDI-estra")
#plt.hist(samples_RWMH[0,:], bins=binsx, density=True)
plt.hist(samples_augALDIone[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.subplot(2,3,5)
plt.title("ALDI-est")
plt.hist(samples_augALDI[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.subplot(2,3,3)
plt.title("CBS")
plt.hist(samples_CBS[0,:], bins=binsx, density=True)
plt.plot(u0s, marg_over_y)
plt.xlim([xmin,xmax])
plt.tight_layout()