import numpy as np
class SqExp:
    def __init__(self):
        return
    def K(self,hyp,D):
        sf2 = np.exp(hyp[0]*2)
        return sf2*np.exp(-np.square(D))
    def hyp(self,D):
        return 1
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return 0
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return


