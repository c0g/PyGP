import numpy as np
from scipy import stats
class PolyGaussian:
    def __init__(self,dt,s20,s2t):
        self.dt = dt
        self.s20 = s20
        self.s2t = s2t
        self.dt = dt
    def eval(self,x,x0,xdot,tend):
        n = int(np.ceil(tend/self.dt))#We want to evaluate from dt - tend. cast to int after rouding up.
        p = np.zeros(np.shape(x))
        for i in xrange(1,n+1):
            mu = x0 + i*self.dt*xdot
            s2 = self.s20 + self.s2t*(self.dt*i)**2
            p += stats.distributions.norm.pdf(x,loc=mu,scale=s2)
        p /= n
        return(p)
        
if __name__=="__main__":
    1+1