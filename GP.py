import numpy as np
from scipy import optimize as opt

#OBSERVATIONS ARE STORED IN NxD MATRICES. N rows, D columns. Hence, obs are //horizontal//


class GaussianProcess(object):
    def __init__(self, lik, hyp, cov):
        self.cov = cov
        self.lik = lik
        self.hyp = hyp
        self.Ymu = np.array([])
        self.Ys2 = np.array([])
        self.Z = np.array([])
        self.F = np.array([])
        self.Zp = np.array([])
        self.T = np.array([])
        self.K = np.array([])

    def predict(self, Zp):
        Kss = self.cov.K(self.hyp, Zp, Zp)[0]
        Ks = self.cov.K(self.hyp, self.Z, Zp)[0]
        self.Ymu = np.dot(Ks.T, np.linalg.solve(self.K, self.F))
        self.pK = Kss - np.dot(Ks.T, np.linalg.solve(self.K, Ks))
        self.Ys2 = np.diag(self.pK)
        self.Ys2.shape = (np.shape(self.Ys2)[0], 1)
        self.Zp = Zp
        return

    def observe(self, Z, Fobs):
        from IPython.core.debugger import Tracer
        #Tracer.set_trace()
        Knew = self.cov.K(self.hyp, Z, Z)[0]
        if self.K.size != 0:
            Ks = self.cov.K(self.hyp, self.Z, Z)[0]
            self.K = np.hstack((np.vstack((self.K, Ks.T)), np.vstack((Ks, Knew))))
            self.F = np.vstack((self.F, Fobs))
            self.Z = np.vstack((self.Z, Z))
        else:
            self.K = Knew
            self.F = np.array(Fobs)
            self.Z = Z
        return

    def draw(self, Zdraw):
        Kss = self.cov.K(self.hyp, Zdraw, Zdraw)[0]
        Ks = self.cov.K(self.hyp, self.Z, Zdraw)[0]
        Ymu = np.dot(Ks.T, np.linalg.solve(self.K, self.F))
        Ys2 = Kss - np.dot(Ks.T, np.linalg.solve(self.K, Ks))
        return np.random.multivariate_normal(Ymu.flatten(), Ys2)

    def optimise_hyper(self):
        prior = np.eye(len(self.hyp))/10
        #bounds = [(-1,1)]*3
        ret = opt.minimize(self.loglik, self.hyp, jac=self.dloglik, method='CG', args=(self.cov, self.Z, self.Z, self.F, prior))
        self.hyp = ret.x.flatten()
        self.K = self.cov.K(self.hyp, self.Z, self.Z)[0]
        return(ret)

    def loglik(self, hyp, cov, Z1, Z2, obs, prior):
        lik = np.exp(self.lik*2)
        likMat = np.eye(np.shape(Z1)[0]) * lik
        hyp2 = hyp.flatten()
        K = cov.K(hyp2, Z1, Z2)[0]
        C = K + likMat

        try:
            L = np.linalg.cholesky(C)
            Cnt = np.linalg.solve(L.T, np.linalg.solve(L, obs))
        except np.linalg.LinAlgError:
            Cnt = np.linalg.solve(C, obs)
        (s, CLogDet) = np.linalg.slogdet(C)
        ll = - 0.5 * CLogDet - 0.5 * obs.T.dot(Cnt) - np.size(obs)*np.log(2*np.pi)/2
        return -ll.flatten()

    def dloglik(self, hyp, cov, Z1, Z2, obs, prior):
        lik = np.exp(self.lik*2)
        likMat = np.eye(np.shape(Z1)[0]) * lik
        hyp2 = hyp.flatten()
        kDk = cov.K(hyp2, Z1, Z2)
        K = kDk[0]
        C = K + likMat
        try:
            L = np.linalg.cholesky(C)
            Cnt = np.linalg.solve(L.T, np.linalg.solve(L, obs))
        except np.linalg.LinAlgError:
            L = None
            Cnt = np.linalg.solve(C, obs)
        dK = kDk[1]

        drivs = [np.array([])]*len(dK)

        for i, d in enumerate(dK):
            if L is not None:
                CnDCn = np.linalg.solve(L.T, np.linalg.solve(L, d))
            else:
                CnDCn = np.linalg.solve(C, d)
            drivs[i] = (-np.trace(CnDCn) + obs.T.dot(CnDCn).dot(Cnt))/2
        return -np.array(drivs).flatten()

if __name__ == "__main__":
    from cov.SquaredExponentialEuc import SqExpEuc
    from cov.NormalARD import NormalARDLoop, NormalARD
    from matplotlib import pyplot as plt
    x = np.linspace(-10,10,10)
    y = np.linspace(-10,10,10)
    xpred = np.linspace(-10,10, 50)
    ypred = np.linspace(-10,10, 50)
    xpred.shape = ypred.shape = (50,1)
    x.shape = y.shape = (10,1)
    (X,Y) = np.meshgrid(x,y)
    X.shape = Y.shape = (10**2,1)
    (XPred,YPred) = np.meshgrid(xpred,ypred)
    XPred.shape = YPred.shape = (50**2,1)
    Z = np.hstack((X,Y))
    ZPred = np.hstack((XPred,YPred))


    obs = np.exp(-(X-0.5)**2) + np.exp(-((Y-0.5)**2))
    obsBig = np.exp(-(XPred-0.5)**2) + np.exp(-((YPred-0.5)**2))
    cov = NormalARD()
    hyp = np.log(np.array([0.1,0.1]))
    lik = np.log(np.array([0.000001]))
    gp = GaussianProcess(lik,hyp,cov)
    gp.observe(Z,obs)
    gp.predict(ZPred)
    ans = gp.Ymu
    ans.shape = (50,50)
    obsBig.shape = (50,50)
    plt.subplot(3,1,1)
    c = plt.contourf(xpred.flatten(),ypred.flatten(),obsBig,100)
    plt.colorbar(c,extend='both')
    plt.plot(gp.Z[:,0], gp.Z[:,1], 'ro')
    plt.subplot(3,1,2)
    c = plt.contourf(xpred.flatten(),ypred.flatten(),ans,100)
    plt.plot(gp.Z[:,0], gp.Z[:,1], 'or')
    plt.colorbar(c,  extend='both')
    plt.subplot(3,1,3)
    gp.optimise_hyper()
    print(gp.hyp)
    gp.predict(ZPred)
    ans = gp.Ymu
    ans.shape = (50,50)
    c = plt.contourf(xpred.flatten(),ypred.flatten(),ans,100)
    plt.plot(gp.Z[:,0], gp.Z[:,1], 'or')
    plt.colorbar(c, extend='both')
    plt.show()


