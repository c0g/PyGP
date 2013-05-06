import numpy as np
class Noise:
    def __init__(self,noise):
        self.s2 = np.exp(-2*noise)
        return
    def K(self,hyp,X,Z):
        s2=self.s2
        if X is Z:
            K = np.eye(np.shape(X)[0],np.shape(X)[0])*s2
        else:
            K = np.zeros((np.shape(X)[0],np.shape(Z)[0]))
        return K
    def _hyp(self,D):
        return 0
    def hyp(self,D):
        return self._hyp(D)
    
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        
