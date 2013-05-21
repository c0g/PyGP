import numpy as np
from scipy.spatial.distance import cdist


class NormalARDLoop(object):
    def __init__(self):
        self.Sig = None
        return

    def hyp(self, D):
        return D

    def K(self, hyp, Z1, Z2):
        el2 = np.exp(hyp*2)
        d = el2.shape[0]
        Sigma2 = np.diag(el2)
        self.Sigma = Sigma2

        def normal(Z1, Z2):
            scale = 1/np.sqrt(np.linalg.det(Sigma2) * (2.*np.pi)**3.)
            rbf = np.exp(-0.5*(Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2))))
            return rbf * scale
        def dfunc_vec(Z1, Z2):
            el2 = np.exp(hyp*2)
            Sigma2 = np.diag(el2)
            return ((Z1 - Z2)**2/el2 - 1
                )*np.exp(-0.5 * (Z1-Z2).T.dot(np.linalg.solve(Sigma2,(Z1-Z2)))
                )/(4*np.pi**(3./2.)*np.sqrt(np.linalg.det(Sigma2)/2))

        f = np.zeros((Z1.shape[0], Z2.shape[0]))
        df = np.zeros((hyp.shape[0], Z1.shape[0], Z2.shape[0]))
        for r, z1 in enumerate(Z1):
            for c, z2 in enumerate(Z2):
                f[r, c] = normal(z1, z2)
                for i,el in enumerate(el2):
                    df[i, r, c] = dfunc_vec(z1, z2)[i]
        return(f, df)


class NormalARD(object):
    def __init__(self):
        self.Sig = None
        return

    def hyp(self, D):
        return D

    def K(self, hyp, Z1, Z2):
        el2 = np.exp(hyp*2)
        d = el2.shape[0]
        Sigma2 = np.diag(el2)
        self.Sigma = Sigma2
        distNorm = cdist(Z1, Z2, 'seuclidean', V=el2)**2
        distFlat = (Z1[:, np.newaxis] - Z2[np.newaxis, :])**2 / el2 - 1

        def normal(dist):
            scale = 1/np.sqrt(np.linalg.det(Sigma2) * (2.*np.pi)**3.)
            rbf = np.exp(-0.5*dist)
            return rbf * scale

        def dfunc(distFlat, distNorm):
            rbf = (np.exp(-0.5 * distNorm
                )/(4*np.pi**(3./2.)*np.sqrt(np.linalg.det(Sigma2)/ 2)))
            
            return distFlat*rbf[:, :, np.newaxis]
        f = normal(distNorm)
        df = dfunc(distFlat, distNorm)
        df = np.rollaxis(df,2)
        return(f, df)

if __name__ == "__main__":
    from scipy.optimize import check_grad
    D = 3
    hyp = np.array([1., 2., 1.])
    cov = NormalARD()
    covL = NormalARDLoop()
    Z1 = np.array([[2., 1., 3.],
        [4.,5.,6.],
        [1.,6.,7],
        [6., 8., 9],
        [5., 7., 11.,],
        [1.,5.,5.]])
    Z2 = np.array([[1.2, .2, 0.2],
        [1.,2.,3.],
        [1.,5.,12.],
        [1.,3.,5.],
        [5.,7.9,3.]])

    def f(hyp):
        return covL.K(hyp, Z1, Z2)[0].flatten()

    def dfL(hyp):
        d = np.array(covL.K(hyp, Z1, Z2)[1])
        return(d)

    dist = 1e-5
    delt = np.zeros(np.shape(hyp))
    delt[0] = dist/2

    print( ( f(hyp+delt) - f(hyp-delt) ) / dist )


    def f(hyp):
        return cov.K(hyp, Z1, Z2)[0].flatten()

    def df(hyp):
        d = np.array(cov.K(hyp, Z1, Z2)[1])
        return(d)

    print( ( f(hyp+delt) - f(hyp-delt) ) / dist ) 
    print( df(hyp) - dfL(hyp) )
