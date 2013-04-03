import numpy as np

class Matern3:
    def __init__(self):
        return
    def hyp(self,D):
        return 1
    def K(self,hyp,D):
        sf2 = np.exp(hyp[0]*2)
        return sf2*(1+D)*np.exp(-D)
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return 0
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        

class Matern5:
    def __init__(self):
        return
    def K(self,hyp,D):
        sf2 = np.exp(hyp[0])
        return sf2*(1+D+np.square(D)/3)*np.exp(-D)
    def hyp(self,D):
        return 1
    def dK(self,hyp,d,i):
        #Vector of differentials w.r.t all hyp
        return 0
    def chain(self,hyp,d):
        return
        
