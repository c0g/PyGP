import numpy as np
class Plus:
    def __init__(self,cov1,cov2):
        self.cov1 = cov1
        self.cov2 = cov2
        return
    def K(self,hyp,X,Z):
        D = np.shape(X)[1]
        nhyp1 = self.cov1.hyp(D)
        hyp1 = hyp[0:nhyp1]
        hyp2 = hyp[nhyp1:]
        return self.cov1.K(hyp1,X,Z) + self.cov2.K(hyp2,X,Z)
    def _hyp(self,D):
        return 0
    def hyp(self,D):
        return self.cov1.hyp(D) + self.cov2.hyp(D)
    
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        