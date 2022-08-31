import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import math


class RectGrid:
    def __init__(self, x1, x2, Nx, y1, y2, Ny, bv):
        self.points_spatial = np.zeros((0, 2))
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.Nx = Nx
        self.Ny = Ny
        self.xs = np.linspace(x1, x2, Nx, endpoint=True)
        self.ys = np.linspace(y1, y2, Ny, endpoint=True)
        # insert all points on the rectangular grid
        #for k in np.linspace(x1, x2, num=math.ceil((x2 - x1 + dx) / dx), endpoint=True):
        for k in self.xs:
            #for l in np.linspace(y1, y2, num=math.ceil((y2 - y1 + dy) / dy), endpoint=True):
            for l in self.ys:
                #		for k in np.arange(x1, x2, dx):
                #			for l in np.arange(y1, y2, dy):
                #				self.points.append(np.array([k, l]))
                self.points_spatial = np.append(self.points_spatial, [[k, l]], axis=0)

        self.numPoints = len(self.points_spatial)
        # create Delaunay triangulation
        self.tri = Delaunay(self.points_spatial)

        ch = self.tri.convex_hull  # returns list of edges which are on the Boundary

        # first specify which points are on Boundary of domain
        self.isOnBoundary = [False] * len(self.points_spatial)
        for edge in ch:
            for vertex in edge:
                self.isOnBoundary[vertex] = True

        self.pInner = np.stack([p for n, p in enumerate(self.points_spatial) if not self.isOnBoundary[n]])
        self.origIndInner = [int(n) for n, p in enumerate(self.points_spatial) if not self.isOnBoundary[n]]
        pNeum_temp = [p for n, p in enumerate(self.points_spatial) if self.isOnBoundary[n] and bv.ind_Neum(p) == 1]
        if len(pNeum_temp) > 0:
            self.pNeum = np.stack(pNeum_temp)
        else:
            self.pNeum = np.array([])
        self.origIndNeum = [int(n) for n, p in enumerate(self.points_spatial) if self.isOnBoundary[n] and bv.ind_Neum(p) == 1]
        self.pDir = np.stack([p for n, p in enumerate(self.points_spatial) if self.isOnBoundary[n] and bv.ind_Dir(p) == 1])
        self.origIndDir = [int(n) for n, p in enumerate(self.points_spatial) if self.isOnBoundary[n] and bv.ind_Dir(p) == 1]

        self.M1 = len(self.pInner)
        self.M2 = len(self.pNeum)
        self.M3 = len(self.pDir)
        if len(self.pNeum > 0):
            self.points = np.concatenate((self.pInner, self.pNeum, self.pDir), axis=0)
        else:
            self.points = np.concatenate((self.pInner, self.pDir),axis=0)
        self.origInd = np.concatenate((self.origIndInner, self.origIndNeum, self.origIndDir),axis=0)
        self.newInd = np.zeros_like(self.origInd, dtype=np.int)
        for n, ind in enumerate(self.origInd):
            self.newInd[ind] = n
        # repeat the whole trinagulation for points in the correct order
        self.tri = Delaunay(self.points)
        ch = self.tri.convex_hull
        # first specify which points are on Boundary of domain
        self.isOnBoundary = [False] * len(self.points)
        for edge in ch:
            for vertex in edge:
                self.isOnBoundary[vertex] = True

        

    def getIndicesObsPoints(self, listOfPoints):
        l = []
        for n, p in enumerate(self.points):
            for p2 in listOfPoints:
                if np.array_equal(p, p2):
                    l.append(n)
        return l
    
    def orderSpatially(self, lst): # reorders a list of the form (inner, Neumann, Dirichlet) to 1d array in the correct order
        ordered = np.zeros((len(lst),))
        for n, p in enumerate(lst):
            ordered[self.origInd[n]] = p
        return ordered
    
    def orderByGroup(self, lst): # reorders a spatially ordered list to the form (inner, Neumann, Dirichlet)
        ordered = np.zeros((len(lst),))
        for n, p in enumerate(lst):
            ordered[self.newInd[n]] = p
        return ordered
    
    def getInnerAndNeumFromAll(self, lst):
        return lst[0:self.M1+self.M2]
        

    def getPoints(self):
        # returns a tuple of lists of points in the correct order (i.e. first interior points, then Boundary points in a list, respectively)
        pInner = self.points[0:self.M1]#[(p, n) for n, p in enumerate(self.points) if not self.isOnBoundary[n]]
        pNeum = self.points[self.M1:self.M1+self.M2]#[(p, n) for n, p in enumerate(self.points) if self.isOnBoundary[n]]
        pDir = self.points[self.M1+self.M2:]
        return (pInner, pNeum, pDir)

    def showGrid(self):
        plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices.copy())
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.show()
    
    def mergePoints(self, valsInner, valsNeum, valsDir):
        vals = np.zeros((self.M1+self.M2+self.M3,))
        for n, ind in enumerate(self.origIndInner):
            vals[ind] = valsInner[n]
        for n, ind in enumerate(self.origIndNeum):
            vals[ind] = valsNeum[n]
        for n, ind in enumerate(self.origIndDir):
            vals[ind] = valsDir[n]
            
        return self.xs, self.ys, np.reshape(vals, (self.Nx, self.Ny))

    def getInnerPoints(self): #### obsolete??
        return [(p, n) for n, p in enumerate(self.points) if not self.isOnBoundary[n]]

    def getBoundaryPointsByFnc(self, indfnc): #### obsolete??
        # returns a list of points p on the Boundary specified by the property indfnc(p) >= 1
        return [(p, n) for n, p in enumerate(self.points) if self.isOnBoundary[n] and indfnc(p) >= 1]


if __name__ == "__main__":
    x1 = 0
    x2 = 1
    dx = 0.25
    y1 = 1
    y2 = 3
    dy = 0.7
    gr = RectGrid(x1, x2, dx, y1, y2, dy)


    #	gr.showGrid()
    def gamma1fnc(vec):
        if vec[0] == 1:
            return 1
        else:
            return 0


    def gamma2fnc(vec):
        return 1 - gamma1fnc(vec)


    #print(gr.getBoundaryPointsByFnc(gamma1fnc))
    #print(gr.getBoundaryPointsByFnc(gamma2fnc))
