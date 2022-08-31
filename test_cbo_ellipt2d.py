# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:55:35 2022

@author: Philipp
"""

import numpy as np
import numpy.ma as ma
from math import sin, pi, cos
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grid import RectGrid

from math import e, log10
import scipy.io as spio

from fwd_multellipt import *
from matplotlib import ticker

import time
from scipy.optimize import minimize

from grad_inference import *
from cbo import *

np.random.seed(5)

plt.ion()
plt.show()


# parameters for multi-Gauss
alpha = 1 # prior regularity


N_opt = 25000 # number of iterations for FISTA
s = 1.0
kappa = 100


sigNoise = 1e-6 # noise standard deviation (redundant parameter, don't change)

N_fourier = 31 # 31

save_file = True

tol = 1e-8


#%% Setting
def ind_dir(vec):
    if (vec[0] < tol or vec[0] > 1-tol or vec[1] > 1-tol) and not (vec[1] < tol):
        return 1
    else:  
        return 0

def ind_neum(vec):
    if vec[1] < tol:
        return 1
    else:
        return 0

def g_dir(vec):
    return vec[:,0]*0.0#-np.cos(2*pi*vec[:,1]) + vec[:,0]**2# + (1 + vec[:,0]**2 + 2 * vec[:,1] ** 2)




g_neum = lambda vec: 0

bv = BoundaryValues(ind_Dir=ind_dir, ind_Neum=ind_neum, g_Dir=g_dir, g_Neum=g_neum)


def coeff_u_01(vec):
    bg = (np.sin(3*vec[0])+np.sin(2*vec[0]))*0.5/log10(e)
    if (vec[0]-1)**2 + vec[1]**2 <= 0.65**2 and (vec[0]-1)**2 + vec[1]**2 > 0.45**2 and vec[0] < 0.85:
        return bg -2/log10(e)
    elif vec[0]-0.2*vec[1] < 0.345 and vec[0]-0.2*vec[1] >= 0.222 and vec[1] >= 0.625:
        return bg - 1.5/log10(e)
    else:
        return bg

def coeff_f(vec, pos, sigx, sigy, strength):
    #return -2.0 - 4*pi**2*np.cos(2*pi*vec[:,1])
    return strength*np.exp(-((vec[:,0]-pos[0])**2)/(2*sigx**2)-((vec[:,1]-pos[1])**2)/(2*sigy**2))

N = 5
MM = 2**N
rg = RectGrid(0, 160.0, MM, 0, 76, MM, bv)
ext = [rg.x1, rg.x2, rg.y1, rg.y2]
#strat_vec = np.flipud(spio.loadmat("stratography.mat", squeeze_me=True)["B"].T.flatten())
strat_vec = (np.fliplr(spio.loadmat("stratography.mat", squeeze_me=True)["B"].T)).flatten()
#strat_symm = (np.fliplr(np.reshape(strat_vec, (MM, MM))).flatten())
strat_symm = ((np.reshape(strat_vec, (MM, MM))).flatten())
stratography = rg.orderByGroup(strat_symm).astype(int)

#umapping = np.random.normal(0, 1, (18,)) # random permeabilities
# umapping = np.random.choice([-2,-1,0,1,2], 18)
umapping_ind = [4,4,2,4,4,4,4,3,3,3,4,2,4,3,3,1,3,2]  # according to Anneli's model
umapping_vals = [-10, np.log(0.3), np.log(0.2), np.log(0.1), np.log(0.01)]
umapping = [umapping_vals[ind] for ind in umapping_ind]

uTruth = np.array([umapping[s] for s in stratography])
#uTruth += np.random.normal(0,0.1,uTruth.shape)

#test different uTruth
matmodestruth = np.zeros((31,31))
matmodestruth[0,2] = 5.0
#matmodestruth[27,2] = -2.0

#plt.savefig("simulations/p_groundtruth.pdf", bbox_inches = 'tight', pad_inches = 0)

kTruth = np.exp(uTruth)
# plot logpermeability
plt.figure()
logkvals_truth = (np.reshape(rg.orderSpatially(np.log(kTruth)), (rg.Nx, rg.Ny)))
#logkvals_truth = np.flipud(np.reshape(rg.orderSpatially(np.log(kTruth)), (rg.Nx, rg.Ny)))
#im = plt.imshow(np.rot90(logkvals_truth), extent=ext, cmap=plt.cm.viridis, interpolation='none', vmin=-3, vmax=3)
#im = plt.imshow(np.rot90(np.exp(logkvals_truth)), extent=ext, cmap=plt.cm.viridis, interpolation='none', vmin=-0, vmax=0.31)
#plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
#plt.savefig("simulations/u_groundtruth.pdf", bbox_inches = 'tight', pad_inches = 0)


strat_spatial = rg.orderSpatially(stratography).reshape((rg.Nx, rg.Ny))
# plt.figure();plt.imshow(np.rot90(strat_spatial), extent=ext)


plt.figure()
plt.imshow(np.rot90(logkvals_truth), extent=ext, interpolation='none', vmin=-5, vmax=-1)
plt.colorbar(fraction=0.022, pad=0.04)
plt.tight_layout()
plt.savefig("uTruth.pdf")

#%%

strength = 10*8.27e-6/0.102 # siehe Wolfgangs Code
    
    
#source_pos = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
# 1.: actual (real sandbox) locations

measXloc = np.array([27.5, 48, 68.4, 93.1, 114, 134])

measYloc = np.flipud(np.array([8.33, 16.5, 24.8, 33, 41.3, 49.6, 57.8, 66.1]))
Xlocmg, Ylocmg = np.meshgrid(measXloc, measYloc)
XX = Xlocmg.flatten()
YY = Ylocmg.flatten()
well_pos_data = np.array([[xx, yy] for xx, yy in zip(XX, YY)])

# 2.: closest indices to real sandbox locations
well_pos_grid_ind = np.zeros((well_pos_data.shape[0],), dtype=np.int32)

ipts, npts, dpts = rg.getPoints()
for rowind in range(well_pos_data.shape[0]):
    spos = well_pos_data[rowind, :]
    well_pos_grid_ind[rowind] = int(np.argmin(np.sum((ipts-spos)**2, axis=1)))

# 3.: measurement location compatible with grid
well_pos_grid = ipts[well_pos_grid_ind]

source_ind =  [1, 4, 13, 16, 31, 34, 43, 46]
source_pos = well_pos_grid[source_ind]
#indobs = well_pos_grid_ind[source_ind]

# make individual observation by dropping the source position
#indobslist = [np.concatenate((indobs[0:kk], indobs[kk+1:])) for kk in range(len(source_ind))]
indobslist = []
for nn, ind in enumerate(source_pos):
    indobslist.append(np.concatenate((well_pos_grid_ind[0:source_ind[nn]], well_pos_grid_ind[source_ind[nn]+1:])))

# width of pump is one half of a cell width
sigx = 0.5*(rg.x2-rg.x1)/(MM-1)
sigy = 0.5*(rg.y2-rg.y1)/(MM-1)
ep = []


#ep = [MultEllipticalProblem(rg, kTruth, lambda vec, temp=ind: coeff_f(vec, source_pos[temp, :], sigx, sigy, strength), bv, indobslist[ind]) for ind in range(source_pos.shape[0])]
ep = MultEllipticalProblem(rg, kTruth, [lambda vec, temp=ind: coeff_f(vec, source_pos[temp, :], sigx, sigy, strength) for ind in range(source_pos.shape[0])] , bv, [indobslist[ind] for ind in range(source_pos.shape[0])])  

   


bdpts = np.array(ep.grid.getBoundaryPointsByFnc(ep.boundaryValues.ind_Dir))

xibars = []
obss = []

plt.figure(); plt.ion()

xibars = ep.fwdOp()
noise = [np.random.normal(0, sigNoise, (len(indobslist[kk]),)) for kk in range(source_pos.shape[0])]
obss = [xibars[kk][indobslist[kk]] + noise[kk] for kk in range(len(xibars))]
for kk in range(len(xibars)):
    xibar = xibars[kk]
    obs = obss[kk]
    plt.subplot(4, 2, kk+1)
    ext = [rg.x1, rg.x2, rg.y1, rg.y2]
    pvals = np.reshape(rg.orderSpatially(np.concatenate((xibar, ep._xihat))), (rg.Nx, rg.Ny))
    plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
    ipts, npts, dpts = rg.getPoints()
    freepts = np.concatenate((ipts, npts), axis=0)
    vmin1 = np.min(obs)
    vmin2 = np.min(pvals)
    vmin = min(vmin1, vmin2)
    vmax1 = np.max(obs)
    vmax2 = np.max(pvals)
    vmax = max(vmax1, vmax2)
    v1 = ipts[indobslist[kk], 0]
    v2 = ipts[indobslist[kk], 1]
    plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.axis("off")

#%%


# Ab hier erst spezifisch zu Prior
def evalmodesGrid(modesmat, x, y, rect, modes_fnc=None): # evaluate function on the whole grid given by x \times y where x and y are np.linspace objects
  if not isinstance(x, np.ndarray):
    x = np.array([[x]])
  if not isinstance(y, np.ndarray):
      y = np.array([[y]])
  # evaluates fourier space decomposition in state space
  N = modesmat.shape[0]
  maxMode = N//2
  freqs = np.reshape(np.linspace(1, maxMode, N//2), (-1, 1))
  M = len(x)
  if modes_fnc is None:
      phi_mat = np.zeros((M, M, N, N))
      X, Y = np.meshgrid(x, y)
      Xprime = (X-rect.x1)/(rect.x2-rect.x1)
      Yprime = (Y-rect.y1)/(rect.y2-rect.y1)
      for k in range(N):
          for l in range(N):
              if k == 0 and l == 0:
                  phi_mat[:, :, 0, 0] = np.ones((M,M))
              elif k == 0 and l > 0 and l <= maxMode:
                  phi_mat[:, :, k, l] = np.cos(l*2*pi*Xprime)
              elif k == 0 and l > 0 and l > maxMode:
                  phi_mat[:, :, k, l] = np.sin((l-maxMode)*2*pi*Xprime)
              elif k > 0 and k <= maxMode and l == 0:
                  phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)
              elif k > 0 and k > maxMode and l == 0:
                  phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)
              elif k > 0 and l > 0:
                  if k <= maxMode and l <= maxMode:
                      phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)*np.cos(l*2*pi*Xprime)
                  elif k <= maxMode and l > maxMode:
                      phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)*np.sin((l-maxMode)*2*pi*Xprime)
                  elif k > maxMode and l <= maxMode:
                      phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)*np.cos(l*2*pi*Xprime)
                  else:
                      phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)*np.sin((l-maxMode)*2*pi*Xprime)
      modes_fnc = phi_mat
  mm = np.reshape(modesmat, (1, 1, N, N))
  mm = np.tile(mm, (M, M, 1, 1))
  temp = mm*modes_fnc
  return np.sum(temp, (2,3))
  
def evalmodesVec(modesmat, x, y, rect, retMat = False): # evaluate function at positions (x0,y0), (x1,y1), ...
    # input: x, y = x0, y0 or
    #             x, y = np.array([x0, x1, ... , x_(M-1)]), np.array([y0, y1, ... , y_(M-1)])

    if (isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and len(x) == 1)): # easy case (just one pair of points)
        N = modesmat.shape[0]
        maxMode = N//2
        freqs = np.reshape(np.linspace(1, maxMode, N//2), (-1, 1))
        phi_mat = np.zeros((N, N))
        
        xprime = (x-rect.x1)/(rect.x2-rect.x1)
        yprime = (y-rect.y1)/(rect.y2-rect.y1)
        for k in range(N):
            for l in range(N):
                if k == 0 and l == 0:
                    phi_mat[0, 0] = 1
                elif k == 0 and l > 0 and l <= maxMode:
                    phi_mat[k, l] = np.cos(l*2*pi*xprime)
                elif k == 0 and l > 0 and l > maxMode:
                    phi_mat[k, l] = np.sin((l-maxMode)*2*pi*xprime)
                elif k > 0 and k <= maxMode and l == 0:
                    phi_mat[k, l] = np.cos(k*2*pi*yprime)
                elif k > 0 and k > maxMode and l == 0:
                    phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)
                elif k > 0 and l > 0:
                    if k <= maxMode and l <= maxMode:
                        phi_mat[k, l] = np.cos(k*2*pi*yprime)*np.cos(l*2*pi*xprime)
                    elif k <= maxMode and l > maxMode:
                        phi_mat[k, l] = np.cos(k*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
                    elif k > maxMode and l <= maxMode:
                        phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.cos(l*2*pi*xprime)
                    else:
                        phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
        temp = modesmat*phi_mat
        return np.sum(temp)
    else:    # hard case: x and y are proper lists
        # evaluates fourier space decomposition in state space
        N = modesmat.shape[0]
        maxMode = N//2
        freqs = np.reshape(np.linspace(1, maxMode, N//2), (-1, 1))
        #x = np.reshape(x, (1, -1))
        M = x.shape[0]
        phi_mat = np.zeros((M, N, N))
        xprime = (x-rect.x1)/(rect.x2-rect.x1)
        yprime = (y-rect.y1)/(rect.y2-rect.y1)
        for k in range(N):
            for l in range(N):
                if k == 0 and l == 0:
                    phi_mat[:, 0, 0] = np.ones((M,))
                elif k == 0 and l > 0 and l <= maxMode:
                    phi_mat[:, k, l] = np.cos(l*2*pi*xprime)
                elif k == 0 and l > 0 and l > maxMode:
                    phi_mat[:, k, l] = np.sin((l-maxMode)*2*pi*xprime)
                elif k > 0 and k <= maxMode and l == 0:
                    phi_mat[:, k, l] = np.cos(k*2*pi*yprime)
                elif k > 0 and k > maxMode and l == 0:
                    phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)
                elif k > 0 and l > 0:
                    if k <= maxMode and l <= maxMode:
                        phi_mat[:, k, l] = np.cos(k*2*pi*yprime)*np.cos(l*2*pi*xprime)
                    elif k <= maxMode and l > maxMode:
                        phi_mat[:, k, l] = np.cos(k*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
                    elif k > maxMode and l <= maxMode:
                        phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.cos(l*2*pi*xprime)
                    else:
                        phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
        if retMat:
            return phi_mat
        mm = np.reshape(modesmat, (1, N, N))
        mm = np.tile(mm, (M, 1, 1))
        temp = mm*phi_mat
        return np.sum(temp, (1,2))
  


def computeScaleMatExp(N, alpha):
    T1 = (alpha*np.arange(0, N//2+1)).reshape((1,-1))
    T2= (alpha*np.arange(1, N//2+1)).reshape((1,-1))
    T3 = (10*alpha*np.arange(0, N//2+1)).reshape((1,-1))
    T4= (10*alpha*np.arange(1, N//2+1)).reshape((1,-1))
    T = np.hstack((T1**2, T2**2))
    T_ = np.hstack((T3**2, T4**2))
    T = np.exp(-np.sqrt(T + T_.T))
    scale = T
    # T1 = np.exp(-alpha*np.arange(0, N//2+1)).reshape((1,-1))
    # T2 = np.exp(-alpha*np.arange(1, N//2+1)).reshape((1,-1))
    # T3 = np.exp(-10*alpha*np.arange(0, N//2+1)).reshape((1,-1))
    # T4 = np.exp(-10*alpha*np.arange(1, N//2+1)).reshape((1,-1))
    # T = np.hstack((T1,T2))
    # T_ = np.hstack((T3,T4))
    # T = T + T_.T
    # scale = T
    return scale



def evalmodes(modes_mat):
    return evalmodesGrid(modes_mat, np.linspace(rg.x1, rg.x2, MM), np.linspace(rg.y1, rg.y2, MM), rg)




def q_onfourier(u_fc_unp, obss): # expects u_fc to be of shape 961 (unpacked)
    u_fc_packed = u_fc_unp.reshape((N_fourier, N_fourier))
    u=rg.orderByGroup((evalmodesGrid(u_fc_packed, rg.xs, rg.ys, rg)).flatten())
    q = ep.qMisfitLogPermeability(u, obss)
    return q

def q_dq_onfourier(u_fc_unp, obss): # expects u_fc to be of shape 961 (unpacked)
    u_fc_packed = u_fc_unp.reshape((N_fourier, N_fourier))
    u=rg.orderByGroup((evalmodesGrid(u_fc_packed, rg.xs, rg.ys, rg)).flatten())
    q, dq = ep.q_dqMisfitLogPermeability(u, obss)
    #dq = np.zeros_like(u_fc_unp)
    xs=rg.xs
    ys=rg.ys
    XX, YY = np.meshgrid(xs, ys)
    X = XX.flatten()
    Y = YY.flatten()
    mat = (evalmodesVec(u_fc_packed,X, Y, rg, retMat = True))
    dq_fourier = np.tensordot(mat, rg.orderSpatially(dq), axes=(0,0))
    return q, dq_fourier.flatten()


#%% test Multigauss
betas = np.array([1.0e-19])
alphas = [0.01]#np.array([0.002, 0.005, 0.01, 0.02, 0.05])
#alphas = np.array([4,6,8])
recons = []

params = []
qvals = []



#%%
n_param = 0
for alpha in alphas:
    n_param += 1
    for beta in betas:    

        setting_str = "scale10_alpha_" + str(alpha) + "_beta_" + str(beta)
        setting_str = str(n_param)
        print(setting_str)
        S = computeScaleMatExp(N_fourier, alpha)
        
        coeff = (S**(-1)).flatten()
        
        
        
        I = lambda x: q_onfourier(x, obss) + beta/2*(x@(coeff*x))
        DI = lambda x: q_dq_onfourier(x, obss)[1] + beta*coeff*x
        
        I_tilde = lambda x: q_onfourier(coeff**(-1)*x, obss) + beta/2*(x@x)
        DI_tilde = lambda x: coeff**(-1)*q_dq_onfourier(coeff**(-1)*x, obss)[1] + beta*x
        
        norm = lambda x: beta/2*(x@(coeff*x))
        
        
        from scipy.optimize import minimize
        u0 = np.zeros((N_fourier**2,))
    
        #test I and DI
        """u0_fc_unp = u0
        u0_fc_unp = uOpt_fc
        I0 = I(u0_fc_unp)
        DI0 = DI(u0_fc_unp)
        
        vec_fc = np.random.multivariate_normal(np.zeros_like(u0_fc_unp),0.000001*np.diag((S**2).flatten()))
        rs = np.linspace(-1,1,21)
        
        Is = np.array([I(u0_fc_unp+r*vec_fc) for r in rs])
        D = np.dot(DI0,vec_fc)
        
        plt.figure();
        plt.plot(rs, Is, '.-')
        plt.ion()
        plt.plot(rs, I0 + rs*D, '.-')
        
        plt.figure()
        plt.plot(rs, Is-(I0 + rs*D), '.-')"""
        
        #test I and DI
        """u0_fc_unp = u0
        u0_fc_unp = uOpt_fc
        I0 = I_tilde(u0_fc_unp)
        DI0 = DI_tilde(u0_fc_unp)
        
        vec_fc = np.random.multivariate_normal(np.zeros_like(u0_fc_unp),0.000001*np.diag((S**2).flatten()))
        rs = np.linspace(-1,1,21)
        
        Is = np.array([I_tilde(u0_fc_unp+r*vec_fc) for r in rs])
        D = np.dot(DI0,vec_fc)
        
        plt.figure();
        plt.plot(rs, Is, '.-')
        plt.ion()
        plt.plot(rs, I0 + rs*D, '.-')
        
        plt.figure()
        plt.plot(rs, Is-(I0 + rs*D), '.-')"""
                
                
        #u0 = uTruth_fc_unp + np.random.normal(0,0.001,((31**2)))
        
        n_it=0
        def callback(xk):
            global n_it
            print(f"iteration {n_it}. value={I_tilde(xk)}")
            n_it += 1
        #result = minimize(I, u0, method='BFGS', jac=DI, options={'gtol': 5e-7, 'disp': True})
        result = minimize(I_tilde, u0, method='CG', jac=DI_tilde, options={'maxiter': 150, 'gtol': 1e-10, 'disp': True}, callback=callback)
        uOpt_fc = coeff**(-1)*(result.x)
        
        #plt.figure();
        #plt.plot(uOpt_fc);
        
        recons.append(uOpt_fc)
        params.append([alpha, beta])
        qvals.append(q_onfourier(uOpt_fc, obss))
        uOpt = rg.orderByGroup(evalmodesGrid(uOpt_fc.reshape((N_fourier, N_fourier)), rg.xs, rg.ys, rg).flatten())
        
        kOpt = np.exp(uOpt)
        plt.figure(); plt.ion()
    
        ep.set_coeff_k(kOpt, True)
        xibars = ep.fwdOp()
    
        for kk in range(len(xibars)):
            xibar = xibars[kk]
            obs = obss[kk]
            plt.subplot(4, 2, kk+1)
            ext = [rg.x1, rg.x2, rg.y1, rg.y2]
            pvals = np.reshape(rg.orderSpatially(np.concatenate((xibar, ep._xihat))), (rg.Nx, rg.Ny))
            plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
            ipts, npts, dpts = rg.getPoints()
            freepts = np.concatenate((ipts, npts), axis=0)
            vmin1 = np.min(obs)
            vmin2 = np.min(pvals)
            vmin = min(vmin1, vmin2)
            vmax1 = np.max(obs)
            vmax2 = np.max(pvals)
            vmax = max(vmax1, vmax2)
            v1 = ipts[indobslist[kk], 0]
            v2 = ipts[indobslist[kk], 1]
            plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
            cb = plt.colorbar()
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
            plt.axis("off")
    
        # if save_file:
        #     plt.savefig("simulations_gauss/p_MAP" + setting_str + ".pdf", bbox_inches = 'tight', pad_inches = 0)
    
        plt.figure()
        logkvals = np.reshape(rg.orderSpatially(np.log(kOpt)), (rg.Nx, rg.Ny))
        im = plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none', vmin=-5, vmax=-1)
        
        #im = plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
        plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
        
        setting_str = f"logperm_MAP_alpha={alpha}.pdf"
        if save_file:
            plt.savefig("simulations_gauss/u_MAP" + setting_str, bbox_inches = 'tight', pad_inches = 0)
        
        # plt.figure()
        # discretized = sortInBins(np.rot90(logkvals), 3)
        # #im = plt.imshow(discretized, extent=ext, cmap=plt.cm.viridis, interpolation='none', vmin=-3, vmax=3)
        # im = plt.imshow(discretized, extent=ext, cmap=plt.cm.viridis, interpolation='none')
        # plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
        # if save_file:
        #     plt.savefig("simulations_gauss/binned" + setting_str + ".pdf",bbox_inches='tight', pad_inches=0)
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(np.rot90(logkvals), extent=ext)
        plt.subplot(2,1,2)
        plt.imshow(np.rot90(logkvals_truth), extent=ext)
        
        #%% test cbo
        # global switch: gradient-based yes or no?
        use_grad = True
        onlymean = True # compute gradient only in mean?
        
        use_truegrad = False # use true gradient instead of inferred Bayesian gradient?
        component_noise = True
        orthogonal_noise = False
        
        d = N_fourier**2
        
        T = 1.0
        tau = 0.005
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
        params["kappa"] = 1.0 # coefficient of gradient drift term
        params["avar"] = 0.0
        
        N_ens = 5 # size of ensemble
        J = N_ens

        
        
        logweightfnc = lambda u: -alpha*I_tilde(u)
        
        # inferGradientAndHess(xs, vs, hessian = True, ind=0, retOnlyMatrix=False)
        
        
        # initial ensemble
        us0_global = np.random.normal(0,1,(d,J)) #np.random.uniform(-4,-1,(d,J)) # hard for functions with minimum at (0,0)
        #us0_global = np.array([[-4,-1.5],[-4,-2.5]])
        #us0_global = np.random.uniform(-2,2,(2,J)) # hard for Himmelblau
        
        
        # list of ensemble particles for each iteration
        us_list = np.zeros(( d, J, N)) # previously: N, d, j
        us_list[:, :, 0] = us0_global
        w_mean = np.zeros((d,N))
        w_mean[:, 0] = weighted_mean(us0_global, lambda u: -params["alpha"]*Phi(u),allow_vectorized_input=False)
        
        
        
        
        for i in range(N-1):
            #us_EKI[i+1, :] = EKI_step(us_EKI[i, :])    
            w_mean[:,i] = weighted_mean(us_list[:, :,i], lambda u: -params["alpha"]*I_tilde(u),allow_vectorized_input=False)
            us_list[:,:,i+1] = step_gradCBO(us_list[:, :, i], Phi, params,tau=tau, maxstep=None, noise="component",allow_vectorized_input=False)
            
        w_mean[:,N-1]  = weighted_mean(us_list[:, :,N-1], lambda u: -params["alpha"]*Phi(u))
        
        #result = minimize(I_tilde, u0, method='CG', jac=DI_tilde, options={'maxiter': 150, 'gtol': 1e-10, 'disp': True}, callback=callback)
        uOpt_fc_cbo = coeff**(-1)*(w_mean[:,N-1])
        
        #plt.figure();
        #plt.plot(uOpt_fc);
        
        recons.append(uOpt_fc_cbo)
        params.append([alpha, beta])
        qvals.append(q_onfourier(uOpt_fc_cbo, obss))
        uOpt_cbo = rg.orderByGroup(evalmodesGrid(uOpt_fc_cbo.reshape((N_fourier, N_fourier)), rg.xs, rg.ys, rg).flatten())
        
        kOpt_cbo = np.exp(uOpt_cbo)
        plt.figure(); plt.ion()
    
        ep.set_coeff_k(kOpt_cbo, True)
        xibars = ep.fwdOp()
        for kk in range(len(xibars)):
            xibar = xibars[kk]
            obs = obss[kk]
            plt.subplot(4, 2, kk+1)
            ext = [rg.x1, rg.x2, rg.y1, rg.y2]
            pvals = np.reshape(rg.orderSpatially(np.concatenate((xibar, ep._xihat))), (rg.Nx, rg.Ny))
            plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
            ipts, npts, dpts = rg.getPoints()
            freepts = np.concatenate((ipts, npts), axis=0)
            vmin1 = np.min(obs)
            vmin2 = np.min(pvals)
            vmin = min(vmin1, vmin2)
            vmax1 = np.max(obs)
            vmax2 = np.max(pvals)
            vmax = max(vmax1, vmax2)
            v1 = ipts[indobslist[kk], 0]
            v2 = ipts[indobslist[kk], 1]
            plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
            cb = plt.colorbar()
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
            plt.axis("off")
        
        plt.figure()
        logkvals_cbo = np.reshape(rg.orderSpatially(np.log(kOpt_cbo)), (rg.Nx, rg.Ny))
        im = plt.imshow(np.rot90(logkvals_cbo), extent=ext, cmap=plt.cm.viridis, interpolation='none', vmin=-5, vmax=-1)
        
        #im = plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
        plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
        

