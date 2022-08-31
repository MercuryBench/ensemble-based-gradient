import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grid import RectGrid

from math import e, log10

import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import bicg

import warnings

class BoundaryValues:
    def __init__(self, ind_Dir = (lambda x: 0), ind_Neum = (lambda x: 0), g_Dir = (lambda x:0), g_Neum = (lambda x:0)):
        self.ind_Dir = ind_Dir
        self.ind_Neum = ind_Neum
        self.g_Dir = g_Dir
        self.g_Neum = g_Neum
	

class MultEllipticalProblem:
	# solves -div(coeff_k * grad(p)) = coeff_f with boundaryValues for p
	# for a given list of coeff_f
    def __init__(self, rectGrid, coeff_k, list_coeff_f, boundaryValues, indObslist):
        self.grid = rectGrid
        self._coeff_k = coeff_k
        self.list_coeff_f = list_coeff_f
        self._numRHS = len(list_coeff_f)
        self.boundaryValues = boundaryValues
        self.indObslist = indObslist
        
        # cached numpy arrays for solving PDEs. Not to be changed explicitly.
        self.cachedMatrix = False
        self._Abar = None
        self._Ahat = None
        self._xihat = None
        
        
        if coeff_k is not None:
        	self.assembleLHS()
        
        # assemble RHS vector cache from list_coeff_f
        p_inner, p_neum, p_dir = self.grid.getPoints()

        M1 = len(p_inner)
        M2 = len(p_neum)
        M3 = len(p_dir)
        M = M1+M2+M3
        qs = [np.zeros((M,)) for n in range(len(self.list_coeff_f))]
        for n in range(len(qs)):
            for simplex in self.grid.tri.simplices:
                ptsLocal = self.grid.points[simplex]
                B = np.array([ptsLocal[1, :] - ptsLocal[0, :], ptsLocal[2, :] - ptsLocal[0, :]]).T
                detBAbs = abs(np.linalg.det(B))		        

                qtilde = self.list_coeff_f[n](self.grid.points[simplex])/6*detBAbs
                for m, i in enumerate(simplex):
                    qs[n][i] = qs[n][i] + qtilde[m]
        self._qbars = [q[0:M1+M2] for q in qs]  # this is fixed for variable k and should never change 
        self._xihat = self.boundaryValues.g_Dir(p_dir) # this is fixed for variable k and should never change 
        
    
    # removes all traces of explicit k
    def clearCache(self):
        self._coeff_k = None
        self.cachedMatrix = False
        self._Abar = None
        self._Ahat = None
    
    def set_coeff_k(self, coeff_k, assembleMatrix=False):
        self._coeff_k = coeff_k

        # remove cache
        self.cachedMatrix = False 
        self._Abar = None
        self._Ahat = None


        # if necessary, recompute cache
        if assembleMatrix:
            self.assembleLHS()
    
    def fwdOp(self):
        # solve PDE for p, given coeff_k, for all coeff_f in list_coeff_f
        xibars = []
        
        if self._coeff_k is None:
            raise Exception("No coeff_k found while trying to execute fwdOp")
        
        if self.cachedMatrix == False:
            warnings.warn("fwdOp is being executed, but no precomputed cache was found. Computing cache now.")
            self.assembleLHS(self._coeff_k)		
        
        Abar = self._Abar
        Ahat = self._Ahat
        xihat = self._xihat
        for qbar in self._qbars:
            bb = qbar - Ahat.dot(xihat)        
            xibar = bicg(Abar, bb, tol=1e-9)[0]
            xibars.append(xibar)
        return xibars
    
    
    def assembleLHS(self):
        coeff_k = self._coeff_k
        p_inner, p_neum, p_dir = self.grid.getPoints()

        S1 = 0.5*np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        S2 = 0.5*np.array([[2, -1, -1], [-1, 0, 1], [-1, 1, 0]])
        S3 = 0.5*np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

        M1 = len(p_inner)
        M2 = len(p_neum)
        M3 = len(p_dir)
        M = M1+M2+M3

        
        list_row_Abar = [] #
        list_col_Abar = [] #
        list_entries_Abar = [] #
        list_row_Ahat = [] #
        list_col_Ahat = [] #
        list_entries_Ahat = [] #
        for simplex in self.grid.tri.simplices:
            ptsLocal = self.grid.points[simplex]
            B = np.array([ptsLocal[1, :] - ptsLocal[0, :], ptsLocal[2, :] - ptsLocal[0, :]]).T
            b1 = B[:, 0]
            b2 = B[:, 1]
            detBAbs = abs(np.linalg.det(B))
            
            gamma1 = 1 / detBAbs * (np.dot(b2.T, b2))
            gamma2 = -1 / detBAbs * (np.dot(b1.T, b2))
            gamma3 = 1 / detBAbs * (np.dot(b1.T, b1))
            if isinstance(coeff_k, np.ndarray):
        	    Atilde = (gamma1 * S1 + gamma2 * S2 + gamma3 * S3) * sum(coeff_k[simplex]) / 3
            elif callable(coeff_k):
                Atilde = (gamma1 * S1 + gamma2 * S2 + gamma3 * S3) * sum(map(coeff_k, self.grid.points[simplex])) / 3
            else:
	            raise Exception("type of k is unknown")

            for m, i in enumerate(simplex):
                for n, j in enumerate(simplex):       	        
                    #A[i, j] = A[i, j] + Atilde[m, n]
                    if i < M1+M2:
                    	if j < M1+M2:
                        	list_row_Abar.append(i) #
                        	list_col_Abar.append(j) #
                        	list_entries_Abar.append(Atilde[m, n]) #
                    	else:
                        	list_row_Ahat.append(i) #
                        	list_col_Ahat.append(j-(M1+M2)) #
                        	list_entries_Ahat.append(Atilde[m, n]) #         
        Abar = coo_matrix((list_entries_Abar, (list_row_Abar, list_col_Abar)), shape=(M1+M2,M1+M2)) #
        Ahat = coo_matrix((list_entries_Ahat, (list_row_Ahat, list_col_Ahat)), shape=(M1+M2,M-(M1+M2))) #
        
        
        self.cachedMatrix = True        
        self._Abar = Abar
        self._Ahat = Ahat  	
    
    def assembleGradient(self, xibar, xihat):
        coeff_k = self._coeff_k
        p_inner, p_neum, p_dir = self.grid.getPoints()

        S1 = 0.5*np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        S2 = 0.5*np.array([[2, -1, -1], [-1, 0, 1], [-1, 1, 0]])
        S3 = 0.5*np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

        M1 = len(p_inner)
        M2 = len(p_neum)
        M3 = len(p_dir)
        M = M1+M2+M3

        #DAXi = np.zeros((M, M))  # (i,j) entry = \partial(A xi)_i/(\partial k(a_j)) 
        list_row = []
        list_col = []
        list_entries = []
        xi = np.concatenate((xibar, xihat))
        for simplex in self.grid.tri.simplices:
            ptsLocal = self.grid.points[simplex]
            B = np.array([ptsLocal[1, :] - ptsLocal[0, :], ptsLocal[2, :] - ptsLocal[0, :]]).T
            b1 = B[:, 0]
            b2 = B[:, 1]
            detBAbs = abs(np.linalg.det(B))
            
            gamma1 = 1 / detBAbs * (np.dot(b2.T, b2))
            gamma2 = -1 / detBAbs * (np.dot(b1.T, b2))
            gamma3 = 1 / detBAbs * (np.dot(b1.T, b1))
            
            DAtilde = (gamma1 * S1 + gamma2 * S2 + gamma3 * S3)/3
            DAXitilde = np.dot(DAtilde, xi[simplex])
            for m, i in enumerate(simplex):
                for k, l in enumerate(simplex):
                    #DAXi[i, l] += DAXitilde[m]
                    if i < M1+M2:
                        list_row.append(i)
                        list_col.append(l)
                        list_entries.append(DAXitilde[m])
        
        # now only return relevant part     
        return coo_matrix((list_entries, (list_row, list_col)), shape=(M1+M2, M))
    
    def qMisfit(self, obss, xibars = None, returnSum = True):
        if xibars is None:
            xibars = self.fwdOp()
        
        qs = np.zeros((self._numRHS,))
        for nn, xibar in enumerate(xibars):
            xi = np.concatenate((xibar, self._xihat))
            qs[nn] = 0.5*np.dot(xi[self.indObslist[nn]]-obss[nn], xi[self.indObslist[nn]]-obss[nn])
        if returnSum:
            return np.sum(qs)
        else:
            return qs
    
    def dqMisfit(self, obss, xibars=None, returnSum = True):
        
        if xibars is None:
            xibars = self.fwdOp()
        dqs = []
        for nn, xibar in enumerate(xibars):        
            rhs = np.zeros(xibar.shape)
            rhs[self.indObslist[nn]] = xibar[self.indObslist[nn]] - obss[nn]
            if self.cachedMatrix == False:
                raise Exception("trying to execute dqMisfit, but no cached matrix found")
            lambdastar = bicg(self._Abar.transpose(), -rhs, tol=1e-9)[0]
            Dres = self.assembleGradient(xibar, self._xihat)
            dqs.append(Dres.transpose().dot(lambdastar))
        if returnSum:
            return sum(dqs)
        else:
            return dqs
    
    def qMisfitLogPermeability(self, us, obss, returnSum=True): # returns misfit for given logpermeability
        ks = np.exp(us)
        self.set_coeff_k(ks, assembleMatrix=True)
        return self.qMisfit(obss, returnSum=returnSum)
    
    def q_dqMisfitLogPermeability(self, us, obss, returnSum=True): # returns misfit and gradient of misfit for given logpermeability
        ks = np.exp(us)
        
        self.set_coeff_k(ks, assembleMatrix=True)
        xibars = self.fwdOp()
        q = self.qMisfit(obss, xibars=xibars, returnSum=returnSum)
        dq = self.dqMisfit(obss, xibars=xibars, returnSum=returnSum)
        
        # chain rule correction
        if returnSum:
            dq = dq*ks
        else:
            dq = [dq_i*ks for dq_i in dq]
        return q, dq


    """def plotSolAndPerm(self, ks=None, xi=None, obs=None, dim3=False, onlyk=False):
        if ks is None:
            ks = self._coeff_k
        if xi is None:
            Abar, Ahat, xihat, qbar = self.assembleData(ks)
            bb = qbar - Ahat.dot(xihat)
            
            xibar = bicg(Abar, bb)[0]
            #xibar = np.linalg.solve(Abar, qbar-np.dot(Ahat, xihat))
            xi = np.concatenate((xibar, xihat))
            
        fig = plt.figure(); plt.ion()
        if dim3 == True:        
            ipts, npts, dpts = self.grid.getPoints()
            freepts = np.concatenate((ipts, npts), axis=0)
            M_free = len(freepts)
            if onlyk == False:
                ax = fig.add_subplot(211, projection='3d')
                v1 = freepts[:, 0]
                v2 = freepts[:, 1]
                v3 = xi[0: M_free]
                ax.scatter(v1, v2, v3, zdir='z', s=20, c=xi[0:M_free])
                ax.scatter(dpts[:, 0], dpts[:, 1], xi[M_free:], 'b.')
                ax.set_zlabel("p")
                if obs is not None:
                    ax.plot(ipts[self.indObs, 0], ipts[self.indObs, 1], obs, 'k.')
                ax = fig.add_subplot(212, projection='3d')
            else:
                ax = fig.add_subplot(111, projection='3d')
            pts = np.concatenate((ipts, npts, dpts), axis=0)
            ax.scatter(pts[:, 0], pts[:, 1], np.log(ks), zdir='z', s=20, c=np.log(ks))
            ax.set_zlabel("log(k)")
            plt.show()
        else:
            ext = [self.grid.x1, self.grid.x2, self.grid.y1, self.grid.y2]
            if onlyk == False:
                plt.subplot(211)
                
                pvals = np.reshape(self.grid.orderSpatially(xi), (self.grid.Nx, self.grid.Ny))
                plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
                if obs is not None:
                    ipts, npts, dpts = self.grid.getPoints()
                    freepts = np.concatenate((ipts, npts), axis=0)
                    vmin1 = np.min(obs)
                    vmin2 = np.min(pvals)
                    vmin = min(vmin1, vmin2)
                    vmax1 = np.max(obs)
                    vmax2 = np.max(pvals)
                    vmax = max(vmax1, vmax2)
                    v1 = ipts[self.indObs, 0]
                    v2 = ipts[self.indObs, 1]
                    plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
                plt.colorbar()
                plt.subplot(212)
            logkvals = np.reshape(self.grid.orderSpatially(np.log(ks)), (self.grid.Nx, self.grid.Ny))
            plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
            plt.colorbar()"""

"""if __name__ == "__main__":
    tol = 1e-9
    plt.ion()
    plt.show()
    sigNoise = 0.01
    np.random.seed(1993)
    def ind_dir(vec):
        if vec[0] < tol or vec[1] > 1-tol or vec[1] < tol or vec[1] > 1-tol:
            return 1
        else:
            return 0

    def ind_neum(vec):
	    return 0

    def g_dir(vec):
	    return vec[:, 0]**2 + 1/9*vec[:, 1]**3#-np.cos(2*pi*vec[:,1]) + vec[:,0]**2# + (1 + vec[:,0]**2 + 2 * vec[:,1] ** 2)


    g_neum = lambda vec: 0
    
    def coeff_f(vec):
        return 2 + 2/3*vec[:,1]
    
    
    bv = BoundaryValues(ind_Dir=ind_dir, ind_Neum=ind_neum, g_Dir=g_dir, g_Neum=g_neum)
    NN = 4
    
    Nx = 2**NN #50
    Ny = 2**NN #30
    grid = RectGrid(0, 2, Nx, 2, 3, Ny, bv)
    ind_2d = np.zeros((Nx-2, Ny-2))
    for kk in range(1, Nx-2, 4):
        for ll in range(1, Ny-2, 4):
            ind_2d[kk, ll] = 1

    ind_1d = np.reshape(ind_2d, (-1,))
    indobs = np.nonzero(ind_1d)[0]
    
    
    def coeff_k(vec):
        return 1.0
    kTruth = np.fromiter(map(coeff_k, grid.points), dtype=np.float64)
    
    ep = EllipticalProblem(grid, coeff_k, coeff_f, bv, indobs)
    import time
    t0 = time.time()
    Abar, Ahat, xihat, qbar = ep.assembleData()
    t1 = time.time()
    xibar = ep.fwdOp(kTruth)
    t2 = time.time()
    print("assembly: " + str(t1-t0))
    print("solution: " + str(t2-t1))
    obs = xibar[indobs] + np.random.normal(0, sigNoise, (len(indobs),))
    ep.plotSolAndPerm(kTruth,  np.concatenate((xibar, xihat)), obs, dim3 = False)
    
    xs = np.linspace(0, 2, Nx);
    ys = np.linspace(2,3, Ny);
    XS, YS = np.meshgrid(xs, ys)
    YS = np.flipud(YS)
    ext = [0, 2, 2, 3]
    plt.figure()
    plt.imshow((XS**2+1/9*YS**3), extent=ext, cmap=plt.cm.viridis, interpolation='none')
    
    
    dq = ep.dqMisfit(kTruth, obs)
    
    plt.figure();plt.plot(dq)"""
    
    
    

