import numpy as np
from scipy.spatial.distance import cdist
class SqExpARD:
    def __init__(self):
        return
    def hyp(self,D):
        return 1+D
    def K(self,hyp,Z1,Z2):
        sf2 = np.exp(2*hyp[0])
        ell =  np.exp(hyp[1:])
        weight = np.diag(1/ell)
        K = cdist(Z1.dot(weight), Z2.dot(weight), metric='sqeuclidean')
        f = sf2*np.exp(-K/2)
        df = [2*f] + [ cdist(np.array([Z1[:,i]/el]).T, np.array([Z2[:,i]/el]).T, metric='sqeuclidean') * f for i,el in enumerate(ell)]
        return(f,df)
if __name__ == "__main__":
    from scipy.optimize import check_grad
    D = 2
    hyp = np.array([1.,2.,1.])
    cov = SqExpARD()
    Z1 = np.array([[1,1]])
    Z2 = np.array([[2,2]])
    def f(hyp):  return cov.K(hyp,Z1,Z2)[0].flatten()
    def df(hyp): 
        d = np.array(cov.K(hyp,Z1,Z2)[1]).flatten()
        return(d)
    print("grad check: " + str(check_grad(f,df,hyp)))
        


