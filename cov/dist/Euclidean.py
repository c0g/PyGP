import numpy as np
from scipy.spatial.distance import euclidean 
class EuclideanDist:
    def __init__(self):
        return
    def K(self,hyp,X,Z):
        l= np.exp(hyp[0])
        return euclidean(X,Z)/l
    def hyp(self):
        return 1
    def dK(self,hyp,d):
        #Vector of differentials w.r.t all hyp
        return 0
    def chain(self,hyp,d):
        #We're part of a chain rule differentiation, differential w.r.t. d
        return
        
