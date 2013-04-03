import numpy as np
from PolyGaussian import PolyGaussian

class Agent:
    def __init__(self,x0,v0,dt,s20,s2t,vmax,id):
        self.x = x0
        self.v = v0
        self.vmax = vmax
        self.pg = PolyGaussian(dt,s20,s2t)
        self.id = id
        self.done = False
        self.p = None
        self.g = None
        
    def propose(self,X,f,tend,gMax=None):
        p = self.pg.eval(X, self.x, self.v, tend)
        self.p = p
        g = p*f
        if gMax is not None:
            g = g-gMax
        self.g = g
        pivot = self.find_idx(X)
        left = np.sum(g[:pivot])
        right = np.sum(g[pivot+1:])
        gain = 0
        self.vprop = self.v
        if right > left:
            if self.v < self.vmax:
                self.vprop = self.v + self.pg.s2t*self.pg.dt
            gain = right
        elif left > right:
            if self.v > - self.vmax:
                self.vprop = self.v - self.pg.s2t*self.pg.dt
            gain = left
        else:
            self.done = True
        return(gain,g)
        
    def act(self,dt):
        self.v = self.vprop
        self.x = self.x + self.v*dt
    
    def find_idx(self,X):
        return (np.abs(X-self.x)).argmin()
    
    def __str__(self):
        return('\nAgent ID: ' + str(self.id) + '\nLocation: ' + str(self.x) + '\nSpeed: ' + str(self.v))  
        
        