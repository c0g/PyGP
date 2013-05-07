import numpy as np
from scipy.spatial.distance import cdist
class SqExpEuc:
    def __init__(self):
        return
    def hyp(self,D):
        return 2
    def K(self,hyp,Z1,Z2):
        K = cdist(Z1, Z2, metric='euclidean')
        f = np.exp(2*hyp[1])*np.exp(-K**2*np.exp(-2*hyp[0]))
        df = (2*K**2*np.exp(-2*hyp[0])*np.exp(2*hyp[1])*np.exp(-K**2*np.exp(-2*hyp[0])),
                2*np.exp(2*hyp[1])*np.exp(-K**2*np.exp(-2*hyp[0])))
        return(f,df)
        


