import numpy as np

class Linear:
    def __init__(self):
        return
    def hyp(self,x,z):
        return 0
    def K(self,hyp,x,z):
        s2 = np.exp(hyp[0]*2)
        return s2*np.dot(x,z.T)
        
        
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return 0
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        
        
