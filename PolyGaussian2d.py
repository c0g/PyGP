import numpy as np
class PolyGaussian:
    def __init__(self,dt,s20,s2t):
        self.dt = dt
        self.s20 = s20
        self.s2t = s2t
        self.dt = dt
    def eval(self,X,Y,x0,xdot,tend):
        n = int(np.ceil(tend/self.dt))#We want to evaluate from dt - tend. cast to int after rouding up.
        print(n)
        p = np.zeros((np.shape(X)[0],np.shape(Y)[0]))
        for i in xrange(1,n+1):
            mu = x0 + i*self.dt*xdot
            s2 = self.s20 + self.s2t*(self.dt*i)**2
            p += np.exp(-0.5*(((X-mu[0])**2)/s2 + ((Y-mu[1])**2)/s2 ))/n

        return(p)
        
if __name__=="__main__":
    1+1