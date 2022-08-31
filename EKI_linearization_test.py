# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 20:01:14 2022

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed()

# X = np.array([[0,1,2],[0,1,4]])
# Cuu = 2/3 * np.cov(X)[0,0]
# Cup = 2/3 * np.cov(X)[0,1]
# Cpp = 2/3 * np.cov(X)[1,1]

# xs = X[0,:]
# ps = X[1,:]
# xm = np.mean(xs)
# pm = np.mean(ps)

# plt.figure()
# plt.plot(xs, ps, '.')

# gamma = 0.04

# # blue1= lambda p: xm + Cup/Cpp*(p-pm)

# # ps_plot = np.linspace(-1,4,100)
# xs_plot = np.linspace(0,2,100)
# # plt.plot(ps_plot, blue1(ps_plot))

# blue = lambda x: pm + Cup/Cuu*(x-xm) 
# plt.plot(xs_plot, blue(xs_plot))


# ps_trafo = blue(xs)
# plt.plot(xs, ps_trafo, 'k.')

# X_trafo = np.stack((xs, ps_trafo))
# Cuu_trafo = 2/3 * np.cov(X_trafo)[0,0]
# Cup_trafo = 2/3 * np.cov(X_trafo)[0,1]
# Cpp_trafo = 2/3 * np.cov(X_trafo)[1,1]

# xs_new = xs + Cup*(Cpp + gamma)**(-1)*(0-ps)
# xs_new_trafo = xs + Cup*(Cpp + gamma)**(-1)*(0-ps_trafo)

# correction = 1/3*np.sum((ps-ps_trafo)**2)

# xs_new_trafo_variant = xs + Cup_trafo*(Cpp_trafo + gamma)**(-1)*(0-ps_trafo)

# alpha = 0.25

# xs_new_trafo_variant2 = xs + Cup_trafo*(Cpp_trafo + correction + gamma)**(-1)*(0-ps_trafo + alpha*(ps_trafo-np.mean(ps_trafo)))


# print(f"vanilla: mean={np.mean(xs_new)}, var = {np.cov(xs_new)}")
# print(f"trafo: mean={np.mean(xs_new_trafo)}, var = {np.cov(xs_new_trafo)}")
# print(f"var1: mean={np.mean(xs_new_trafo_variant)}, var = {np.cov(xs_new_trafo_variant)}")
# print(f"var2: mean={np.mean(xs_new_trafo_variant2)}, var = {np.cov(xs_new_trafo_variant2)}")

# plt.plot(xs_new, xs_new**2, '.', label="EK")
# plt.plot(xs_new_trafo, xs_new_trafo**2+ np.random.normal(0,0.1), '.', label="trafo")
# plt.plot(xs_new_trafo_variant, xs_new_trafo_variant**2, '.', label="var")
# plt.plot(xs_new_trafo_variant2, xs_new_trafo_variant2**2, '.', label="var2")
# plt.legend()

sigNoise = 0.2
sigPrior = 2
J = 20
u0 = np.random.normal(0,1.0,(1,J))
u0 = u0 - np.mean(u0)

# u0 = np.array([-3,-2,-1,0,1,2,3])[np.newaxis,:]+np.random.normal(0,0.5,(1,7))
# J = 7

G = lambda u: u**2
y = 1
Phi = lambda u: 0.5/(sigNoise**2)*(G(u)-y)**2
I = lambda u: Phi(u) + 0.5/(sigPrior**2)*np.linalg.norm(u,axis=0)**2

# plt.figure()
# plt.plot(xs_plot, np.exp(-Phi(xs_plot)))

xs_plot = np.linspace(-2,2,200)

#%%

# plain EKI
EKI_plain = True
N_sim = 1000
tau = 0.02
us_list_EKI = np.zeros((1,J,N_sim))
ms_list_EKI = np.zeros((2,N_sim))
cov_list_EKI = np.zeros((3,N_sim))
us_list_EKI[:,:,0] = u0

t1 = time.time()
tau_adaptive = tau



for n in range(N_sim-1):    
    us = us_list_EKI[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    G_us = G_us_unprocessed
    m_G_us = np.mean(G_us, axis=1)[np.newaxis,:]
    ms_list_EKI[0,n] = m_us.flatten()
    ms_list_EKI[1,n] = m_G_us.flatten()
    u_c = us-m_us
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c)[0,0]
    C = np.cov(us)*(J-1)/J#np.mean((us-m_us)**2,axis=1)
    E = np.cov(G_us)*(J-1)/J
    cov_list_EKI[0,n] = C
    cov_list_EKI[1,n] = D
    cov_list_EKI[2,n] = E
    # increment = D@(G_us-y[:,np.newaxis])
    
    
    tau_adaptive = tau
    
    us_list_EKI[:,:,n+1] = us - tau_adaptive*D*(E+sigNoise**2)**(-1)*(G_us-y) #- tau_adaptive/sigPrior**2*C*us

print(f"plain EKI: {time.time()-t1}")


# different variant of EKI
EKI_variant = True
us_list_EKIv = np.zeros((1,J,N_sim))
ms_list_EKIv = np.zeros((2,N_sim))
cov_list_EKIv = np.zeros((3,N_sim))
us_list_EKIv[:,:,0] = u0
    
for n in range(N_sim-1):    
    us = us_list_EKIv[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    G_us = G_us_unprocessed
    m_G_us = np.mean(G_us, axis=1)[np.newaxis,:]
    ms_list_EKIv[0,n] = m_us.flatten()
    ms_list_EKIv[1,n] = m_G_us.flatten()
    u_c = us-m_us
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c)[0,0]
    C = np.cov(us)*(J-1)/J#np.mean((us-m_us)**2,axis=1)
    E = np.cov(G_us)*(J-1)/J
    
    
    interp = lambda x: m_G_us + D/C*(x-m_us)
    G_linear = interp(us)
    Ecal = np.cov(G_linear)*(J-1)/J
    Theta = 1/J*np.sum((G_linear-G_us)**2)
    
    cov_list_EKIv[0,n] = C
    cov_list_EKIv[1,n] = D
    cov_list_EKIv[2,n] = E
    # increment = D@(G_us-y[:,np.newaxis])
    
    
    tau_adaptive = tau
    
    us_list_EKIv[:,:,n+1] = us - tau_adaptive*D*(Ecal+Theta+sigNoise**2)**(-1)*(G_linear-y) \
        #- tau_adaptive*D*(Ecal+Theta+sigNoise**2)**(-1)*(G_us - G_linear)

plt.figure()
plt.plot(us_list_EKI[0,:,:].T)
plt.figure()
plt.plot(us_list_EKIv[0,:,:].T)

fig, ax = plt.subplots(1,1, squeeze=False)

plt.plot(xs_plot, G(xs_plot))
sc = plt.scatter(u0, G(u0))
plt.plot(xs_plot, 0*xs_plot+y)
line = plt.plot([0],[0],'r')[0]


for n in range(N_sim-1):
    
    
    us = us_list_EKI[:,:,n]
    
    interp = lambda x: ms_list_EKI[1,n] + cov_list_EKI[1,n] /cov_list_EKI[0,n]*(x-ms_list_EKI[0,n])
    if n % 20 == 0:
        sc.set_offsets(np.stack((us_list_EKI[0,:,n+1],G(us_list_EKI[0,:,n+1]))).T)
        line.set_xdata(xs_plot)
        line.set_ydata(interp(xs_plot))
        #ax[0,0].set_title(f"t={n}")
        plt.pause(0.1)
        
        

