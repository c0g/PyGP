import numpy as np
from scipy.spatial.distance import cdist
class EuclideanDist:
    def __init__(self,cov):
        self.cov = cov
        return
    def K(self,hyp,Z1,Z2):
        #ipdb.set_trace()
        ell = np.exp(2*hyp[0])
        K = cdist(Z1, Z2, metric='euclidean')/ell
        K = self.cov.K(hyp[1:],K)
        return K   
    def _hyp(self,D):
        return 1
    def hyp(self,D):
        return self._hyp(D) + self.cov.hyp(D)
    
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        
