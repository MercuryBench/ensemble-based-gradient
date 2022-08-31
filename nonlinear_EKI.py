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
from scipy.optimize import minimize

np.random.seed(1)

adaptive_tau = True
backtracking = True

usetruegrad = False

c = 0.5
rho = 0.5

tau = 0.1

alpha = 1.0
J = 7
N_sim = 500

#-----------------------------------
# test case "banana"
y = np.array([0])
d = 2
sigNoise = 0.5
sigPrior = 2
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
# #
# d= 2
# u0 = np.random.normal(0,0.2,(2,J)) + np.array([[-3.0],[102]])

# xmin = -3.75
# xmax = -1.75
# ymin =  100
# ymax =  106

# xobs = np.array([0.25, 0.75])



# p = lambda x, u: np.tensordot(u[1],x,axes=0) + np.tensordot(np.exp(-u[0]),(-x**2/2+x/2), axes=0)
# Dp = lambda x, u: np.stack((-np.tensordot((-x**2/2+x/2),np.exp(-u[0]), axes=0),x[:,np.newaxis]+0*u[1]))
# G = lambda u: p(xobs, u)
# DG = lambda u: Dp(xobs, u)
# Phi = lambda u: 0.5/(sigNoise**2)*np.linalg.norm(G(u)-y, axis=-1)**2
# DPhi = lambda u: 1/sigNoise**2*np.einsum('ijk,kj->ik',DG(u),G(u)-y)


# parameters independent of test case



I = lambda u: Phi(u) + 0.5/(sigPrior**2)*np.linalg.norm(u,axis=0)**2
DI = lambda u: DPhi(u) + 1/sigPrior**2*u

res_opt = minimize(I,np.array([0,0]), tol=1e-11)
x_opt = res_opt.x

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


EKI_grad = False
EKI_gradone = False
EKI_plain = False
GD = False
GD_one = False

#%%

# grad EKI
t1 = time.time()
us_list = np.zeros((2,J,N_sim))
us_list[:,:,0] = u0


EKI_grad = True


tau_adaptive = tau
memory = None
for n in range(N_sim-1):    
    us = us_list[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    if usetruegrad:
        vs = DI(us) ############################################
        v_mean = DI(m_us) ##############################
    else:
        if memory is None:
            ensembleplusmean = np.concatenate((m_us,us),axis=1)
        else:
            ensembleplusmean = np.concatenate((m_us,us,memory),axis=1)
        vs, H = compute_gradients(ensembleplusmean)  
        v_mean = vs[:,0][:,np.newaxis]
        vs = vs[:,1:J+1] # remove gradient of mean 
        
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
    
    # step size control
    if adaptive_tau:
        tau_adaptive = tau/(1e-20+np.linalg.norm(C))
    else:
        tau_adaptive = tau
    if backtracking:
        tau_bt = tau_adaptive
        counter = 10 # at most 10 backtracking steps
        Is_old = I(us)
        Is_new = I(us - tau_bt*(C)@(vs))
        
        while np.sum(Is_new) > np.sum(Is_old - c*tau_bt*vs.T@(C@vs)) and counter > 0:
            #while I(m_us-tau_bt*C@v_mean) > I(m_us)-c*tau_bt*v_mean.T@C@v_mean and counter > 0:
            tau_bt *= rho
            counter -= 1
        tau_adaptive = tau_bt
    
    us_list[:,:,n+1] = us - tau_adaptive*(C)@(vs)
    
    memory = np.copy(us)
             
            
    
    # if adaptivetau:
    #     tau_normscaled = tau/(1e-20+np.linalg.norm(C))#tau/np.linalg.norm(increment)
    # else:
    #     if backtracking:
    #         tau_adaptive = tau_adaptive/rho
    #         tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
    #         while I(m_us-tau_normscaled*C@v_mean) > I(m_us)-c*tau_normscaled*v_mean.T@C@v_mean:#while np.sum(I( us - tau_normscaled*(C)@(vs))) > np.sum(I(us)) - c*tau_normscaled*np.sum(np.diag(vs.T@(C@vs))):
    #             tau_adaptive = rho*tau_adaptive
    #             tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
    #         print(tau_adaptive)    
            
    # us_list[:,:,n+1] = us - tau_normscaled*(C)@(vs)
print(f"grad EKI: {time.time()-t1}")
    
    
#%%
# one grad EKI

EKI_gradone = True


t1 = time.time()
us_list_one = np.zeros((2,J,N_sim))
us_list_one[:,:,0] = u0
memory = None
tau_adaptive = tau
for n in range(N_sim-1):    
    us = us_list_one[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    if memory is None:
        ensembleplusmean = np.concatenate((m_us,us),axis=1)
    else:
        ensembleplusmean = np.concatenate((m_us,us,memory),axis=1)
        
    return_dict= inferGradientAndHess(ensembleplusmean, I(ensembleplusmean), hessian = True, ind=0, additionalvariance=0.0)
    v, H = return_dict['grad'], return_dict['H']
    v_mean = v[:,np.newaxis]
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
    vs = v_mean+H@(u_c)


    if adaptive_tau:
        tau_adaptive = tau/(1e-20+np.linalg.norm(C))
    else:
        tau_adaptive = tau
    if backtracking:
        tau_bt = tau_adaptive
        counter = 10 # at most 10 backtracking steps
        while I(m_us-tau_bt*C@v_mean) > I(m_us)-c*tau_bt*v_mean.T@C@v_mean and counter > 0:
            tau_bt *= rho
            counter -= 1
        tau_adaptive = tau_bt
    
    # us_list[:,:,n+1] = us - tau_adaptive*(C)@(vs)

    # if adaptivetau:
    #     tau_normscaled = tau/(1e-20+np.linalg.norm(C))#tau/np.linalg.norm(increment)
    # else:
    #     if backtracking:
    #         tau_adaptive = tau_adaptive/rho
    #         while np.sum(I( us - tau_adaptive/(1e-20+np.linalg.norm(C))*(C)@(vs))) > np.sum(I(us)) - c*tau_adaptive/(1e-20+np.linalg.norm(C))*np.sum(np.diag(vs.T@(C@vs))):
    #             tau_adaptive = rho*tau_adaptive
    #         tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
    #         print(tau_adaptive)    
    #     else:
    #         tau_normscaled = tau
    us_list_one[:,:,n+1] = us - tau_adaptive*(C)@(vs )
    memory = np.copy(us)
print(f"one grad EKI: {time.time()-t1}")

#%%
# plain EKI
EKI_plain = True
us_list_EKI = np.zeros((2,J,N_sim))
us_list_EKI[:,:,0] = u0

t1 = time.time()
tau_adaptive = tau
for n in range(N_sim-1):    
    us = us_list_EKI[:,:,n]
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
    # increment = D@(G_us-y[:,np.newaxis])
    
    
    
    if adaptive_tau:
        tau_adaptive = tau/(1e-20+np.linalg.norm(C))
    else:
        tau_adaptive = tau
    if backtracking:
        tau_bt = tau_adaptive
        counter = 10 # at most 10 backtracking steps
        Is_old = I(us)
        I_old_mean = I(m_us)
        Is_new = I(us - tau_bt*1/sigNoise**2*D@(G_us-y[:,np.newaxis]) - tau_bt/sigPrior**2*C@us)
        I_new_mean = I(m_us - tau_bt*1/sigNoise**2*D@(m_G_us-y) - tau_bt/sigPrior**2*C@m_us)
        descent = np.linalg.norm(1/sqrt(J)*np.sum(g_c.T@(m_G_us-y), axis=0))**2
        while I_new_mean > I_old_mean - c*tau_bt*descent: #np.sum(Is_new) > np.sum(Is_old)-c*tau_bt*descent and counter > 0: # simplified backtracking
            #while I(m_us-tau_bt*1/sigNoise**2*(D@(G(m_us)-y).T).reshape((d,-1)) -tau_bt/sigPrior**2*C@m_us) >  I(m_us) and counter > 0: # simplified backtracking
            tau_bt *= rho
            counter -= 1
            I_new_mean = I(m_us - tau_bt*1/sigNoise**2*D@(m_G_us-y) - tau_bt/sigPrior**2*C@m_us)
            descent = np.linalg.norm(1/sqrt(J)*np.sum(g_c.T@(m_G_us-y), axis=0))**2
        tau_adaptive = tau_bt
    
    # if adaptivetau:
    #     tau_normscaled = tau/(1e-20+np.linalg.norm(C))#tau/np.linalg.norm(increment)
    # else: 
    #     if backtracking:
    #         # tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
    #         # continue
    #         tau_adaptive = tau_adaptive/rho
    #         tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
            
    #         while I(m_us-tau_normscaled*D@(G(m_us)-y).T-tau_normscaled/sigPrior**2*C@m_us) > I(m_us)-c*tau_normscaled*v_mean.T@(D@(G(m_us)-y).T +1/sigPrior**2*C@m_us):
    #             tau_adaptive = rho*tau_adaptive
    #             tau_normscaled = tau_adaptive/(1e-20+np.linalg.norm(C))
    #         print(tau_adaptive)    
    #     tau_adaptive = tau
    us_list_EKI[:,:,n+1] = us - tau_adaptive*1/sigNoise**2*D@(G_us-y[:,np.newaxis]) - tau_adaptive/sigPrior**2*C@us
print(f"plain EKI: {time.time()-t1}")

#%%
# =============================================================================
# 
# # grad descent
# GD = True
# 
# us_list_GD = np.zeros((2,J,N_sim))
# us_list_GD[:,:,0] = u0
# 
# t1 = time.time()
# 
# tau_adaptive = tau
# memory = None
# for n in range(N_sim-1):    
#     us = us_list_GD[:,:,n]
#     m_us = np.mean(us, axis=1)[:,np.newaxis]
# 
#     if usetruegrad:
#         vs = DI(us) 
#         v_mean = DI(m_us)
#     else:    
#         if memory is None:
#             ensembleplusmean = np.concatenate((m_us,us),axis=1)
#         else:
#             ensembleplusmean = np.concatenate((m_us,us,memory),axis=1)
#         vs, H = compute_gradients(ensembleplusmean)  
#         v_mean = vs[:,0][:,np.newaxis] # gradient in mean for backtracking
#         vs = vs[:,1:J+1] # remove gradient of mean 
#     
#     
#     if adaptive_tau:
#         tau_adaptive = tau #(scaling does not make any sense here) due to missing preconditioner
#     else:
#         tau_adaptive = tau
#     if backtracking:
#         tau_bt = tau_adaptive
#         counter = 10 # at most 10 backtracking steps
#         while I(m_us-tau_bt*v_mean) > I(m_us)-c*tau_bt*v_mean.T@v_mean and counter > 0:
#             tau_bt *= rho
#             counter -= 1
#         tau_adaptive = tau_bt
#     
#     
#     # increment = (vs)
#     # if adaptivetau:
#     #     tau_adaptive = tau#/np.linalg.norm(increment)
#     # else:
#     #     if backtracking:
#     #         tau_adaptive = tau_adaptive/rho
#     #         while I(m_us-tau_adaptive*v_mean) > I(m_us)-c*tau_adaptive*v_mean.T@v_mean:#np.sum(I( us - tau_adaptive*(vs))) > np.sum(I(us)) + c*tau_adaptive*np.sum(np.diag(vs.T@(vs))):
#     #             tau_adaptive = rho*tau_adaptive
#     #         print(tau_adaptive)    
#     us_list_GD[:,:,n+1] = us - tau_adaptive*(vs)
#     memory = np.copy(us)
# print(f"grad desc: {time.time()-t1}")
#     
# =============================================================================

#%%


GD_one = True


# grad descent with extrapolation
us_list_GDextra = np.zeros((2,J,N_sim))
us_list_GDextra[:,:,0] = u0

t1 = time.time()

tau_adaptive = tau
memory = None
for n in range(N_sim-1):    
    us = us_list_GDextra[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    u_c = us-m_us
    if memory is None:
        ensembleplusmean = np.concatenate((m_us,us),axis=1)
    else:
        ensembleplusmean = np.concatenate((m_us,us,memory),axis=1)
    
    return_dict= inferGradientAndHess(ensembleplusmean, I(ensembleplusmean), hessian = True, ind=0, additionalvariance=0.0)
    v, H = return_dict['grad'], return_dict['H']
    v_mean = v[:,np.newaxis]
    vs = v_mean + H@(u_c)
    
    
    
    if adaptive_tau:
        tau_adaptive = tau #(scaling does not make any sense here) due to missing preconditioner
    else:
        tau_adaptive = tau
    if backtracking:
        tau_bt = tau_adaptive
        counter = 10 # at most 10 backtracking steps
        Is_old = I(us)
        Is_new = I(us - tau_bt*v_mean)
        descent = np.einsum("ij,ij->j",vs,vs)
        Is_old - c*tau_bt*descent
        while np.sum(Is_new) > np.sum(Is_old - c*tau_bt*descent):#I(m_us-tau_bt*v_mean) > I(m_us)-c*tau_bt*v_mean.T@v_mean:            
            tau_bt *= rho
            counter -= 1
            Is_new = I(us - tau_bt*v_mean)
        tau_adaptive = tau_bt
    
    # increment = (vs)
    # if adaptivetau:
    #     tau_adaptive = tau#/np.linalg.norm(increment)
    # else:
    #     if backtracking:
    #         tau_adaptive = tau_adaptive/rho
    #         while np.sum(I( us - tau_adaptive*(vs))) > np.sum(I(us)) - c*tau_adaptive*np.sum(np.diag(vs.T@(vs))):
    #             tau_adaptive = rho*tau_adaptive
    #         print(tau_adaptive)    
    us_list_GDextra[:,:,n+1] = us - tau_adaptive*(vs)
    memory = np.copy(us)
print(f"grad desc extra: {time.time()-t1}")
    

#%%


from scipy.optimize import minimize
res = minimize(I, np.array([-3,103]),tol=1e-11)
xopt = res.x

u0s = np.linspace(xmin,xmax,150)
u1s = np.linspace(ymin,ymax,150)
U0, U1 = np.meshgrid(u0s,u1s)
U = np.stack((U0,U1))


if EKI_grad:    
    plt.figure()
    plt.suptitle("grad EKI")
    plt.subplot(3,1,1)
    plt.plot(us_list[0,:,:].T)
    plt.subplot(3,1,2)
    plt.plot(us_list[1,:,:].T)
    plt.subplot(3,1,3)
    plt.semilogy(np.linalg.norm(us_list[:,:,:]-np.mean(us_list,axis=1,keepdims=True),axis=0).T)
    
    
    
    
    plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
    plt.contourf(U0, U1, np.exp(-I(U)), 10)
    plt.title("grad EKI")
    plt.plot(us_list[0,:,:].T, us_list[1,:,:].T, '.-')
    plt.plot(us_list[0,:,-1].T, us_list[1,:,-1], 'ks')
    plt.plot(x_opt[0],x_opt[1], 'rx')
    plt.tight_layout()
    #plt.axis("equal")
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))

if EKI_gradone:
    plt.figure()
    plt.suptitle("grad-one EKI")
    plt.subplot(3,1,1)
    plt.plot(us_list_one[0,:,:].T)
    plt.subplot(3,1,2)
    plt.plot(us_list_one[1,:,:].T)
    plt.subplot(3,1,3)
    plt.semilogy(np.linalg.norm(us_list_one[:,:,:]-np.mean(us_list_one,axis=1,keepdims=True),axis=0).T)
    
    
    plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
    plt.contourf(U0, U1, np.exp(-I(U)), 10)
    plt.title("grad-one EKI")
    plt.plot(us_list_one[0,:,:].T, us_list_one[1,:,:].T, '.-')
    plt.plot(us_list_one[0,:,-1].T, us_list_one[1,:,-1], 'ks')
    plt.plot(x_opt[0],x_opt[1], 'rx')
    plt.tight_layout()




if EKI_plain:
    plt.figure()
    plt.suptitle("plain EKI")
    plt.subplot(3,1,1)
    plt.plot(us_list_EKI[0,:,:].T)
    plt.subplot(3,1,2)
    plt.plot(us_list_EKI[1,:,:].T)
    plt.subplot(3,1,3)
    plt.semilogy(np.linalg.norm(us_list_EKI[:,:,:]-np.mean(us_list_EKI,axis=1,keepdims=True),axis=0).T)
    
    
    plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
    plt.contourf(U0, U1, np.exp(-I(U)), 10)
    plt.title("plain EKI")
    plt.plot(us_list_EKI[0,:,:].T, us_list_EKI[1,:,:].T, '.-')
    plt.plot(us_list_EKI[0,:,-1].T, us_list_EKI[1,:,-1], 'ks')
    plt.plot(x_opt[0],x_opt[1], 'rx')
    plt.tight_layout()

if GD:
    plt.figure()
    plt.suptitle("grad desc")
    plt.subplot(3,1,1)
    plt.plot(us_list_GD[0,:,:].T)
    plt.subplot(3,1,2)
    plt.plot(us_list_GD[1,:,:].T)
    plt.subplot(3,1,3)
    plt.semilogy(np.linalg.norm(us_list_GD[:,:,:]-np.mean(us_list_GD,axis=1,keepdims=True),axis=0).T)
    
    
    plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
    plt.contourf(U0, U1, np.exp(-I(U)), 10)
    plt.title("grad desc")
    plt.plot(us_list_GD[0,:,:].T, us_list_GD[1,:,:].T, '.-')
    plt.plot(us_list_GD[0,:,-1].T, us_list_GD[1,:,-1], 'ks')
    #plt.plot(m_us[0],m_us[1],'rx')
    plt.plot(x_opt[0],x_opt[1], 'rx')
    plt.tight_layout()

if GD_one:
    
    plt.figure()
    plt.suptitle("grad desc extra")
    plt.subplot(3,1,1)
    plt.plot(us_list_GDextra[0,:,:].T)
    plt.subplot(3,1,2)
    plt.plot(us_list_GDextra[1,:,:].T)
    plt.subplot(3,1,3)
    plt.semilogy(np.linalg.norm(us_list_GDextra[:,:,:]-np.mean(us_list_GDextra,axis=1,keepdims=True),axis=0).T)
    
    
    plt.figure()#figsize=(xmax-xmin,0.8*(ymax-ymin)))
    plt.contourf(U0, U1, np.exp(-I(U)), 10)
    plt.title("grad desc extra")
    plt.plot(us_list_GDextra[0,:,:].T, us_list_GDextra[1,:,:].T, '.-')
    plt.plot(us_list_GDextra[0,:,-1].T, us_list_GDextra[1,:,-1], 'ks')
    plt.plot(x_opt[0],x_opt[1], 'rx')
    plt.tight_layout()

# plt.figure(figsize=(5,8))
# plt.subplot(5,1,1)
# plt.semilogy(np.linalg.norm(us_list- xopt[:,np.newaxis,np.newaxis],axis=0).T)
# plt.title("grad EKI")
# plt.subplot(5,1,2)
# plt.semilogy(np.linalg.norm(us_list_one- xopt[:,np.newaxis,np.newaxis],axis=0).T)
# plt.title("grad-one EKI")
# plt.subplot(5,1,3)
# plt.semilogy(np.linalg.norm(us_list_EKI- xopt[:,np.newaxis,np.newaxis],axis=0).T)
# plt.title("plain EKI")
# plt.subplot(5,1,4)
# plt.semilogy(np.linalg.norm(us_list_GD- xopt[:,np.newaxis,np.newaxis],axis=0).T)
# plt.title("Grad desc")
# plt.subplot(5,1,5)
# plt.semilogy(np.linalg.norm(us_list_GDextra- xopt[:,np.newaxis,np.newaxis],axis=0).T)
# plt.title("grad desc ext")


plt.figure()
plt.title("function value")
if EKI_grad:
    plt.semilogy(I(np.mean(us_list,axis=1)).T-res.fun, label="grad EKI")
if EKI_gradone:
    plt.semilogy(I(np.mean(us_list_one,axis=1)).T-res.fun, label="grad-one EKI")
if EKI_plain:
    plt.semilogy(I(np.mean(us_list_EKI,axis=1)).T-res.fun, label="vanilla EKI")
if GD:
    plt.semilogy(I(np.mean(us_list_GD,axis=1)).T-res.fun, label="grad desc")
if GD_one:
    plt.semilogy(I(np.mean(us_list_GDextra,axis=1)).T-res.fun, label="grad desc extr")
plt.legend()

plt.figure()
plt.title("distance to theoretical minimum")
if EKI_grad:
    plt.semilogy(np.mean(np.linalg.norm(us_list- xopt[:,np.newaxis,np.newaxis],axis=0),axis=0), label="EKI grad")
if EKI_gradone:
    plt.semilogy(np.mean(np.linalg.norm(us_list_one- xopt[:,np.newaxis,np.newaxis],axis=0),axis=0), label="EKI one-grad")
if EKI_plain:
    plt.semilogy(np.mean(np.linalg.norm(us_list_EKI- xopt[:,np.newaxis,np.newaxis],axis=0),axis=0), label="EKI")
if GD:
    plt.semilogy(np.mean(np.linalg.norm(us_list_GD- xopt[:,np.newaxis,np.newaxis],axis=0),axis=0), label="GD")
if GD_one:
    plt.semilogy(np.mean(np.linalg.norm(us_list_GDextra- xopt[:,np.newaxis,np.newaxis],axis=0),axis=0), label="GD extra")



plt.legend()