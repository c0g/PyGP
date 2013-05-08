import numpy as np
from util.ltri import ltri
from scipy.spatial.distance import cdist
class MahalanobisDist:
    def __init__(self,cov):
        self.cov = cov
        return
    def K(self,hyp,X,Z):
        D = np.shape(X)[1]
        NHyp = self._hyp(D)
        L = ltri(np.exp(hyp[0:NHyp]))
        VI = np.dot(L,L.T);
        K = cdist(X, Z, metric='mahalanobis',VI=VI)
        K = self.cov.K(hyp[NHyp-1:],K)
        return K   
    def _hyp(self,D):
        return (D+1)*D/2
    def hyp(self,D):
        return self._hyp(D) + self.cov.hyp(D)
    
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        
