import numpy as np
from scipy import optimize as opt


class GaussianProcess():
    def __init__(self,hyp,cov):
        self.cov = cov
        self.hyp = hyp
        self.Ymu = None
        self.Ys2 = None
        self.X = None
        self.Y = None
        self.Z = None
        self.T = None
        self.K = None
        
    def get_Ys2(self,Z):
        Z = np.array([Z])
        Z.shape = (1,1)
        Kss = self.cov.K(self.hyp,Z,Z)
        Ks = self.cov.K(self.hyp,self.X,Z)
        pK = Kss - np.dot(Ks.T,np.linalg.solve(self.K,Ks))
        Ys2 = np.diag(pK)
        Ys2.shape = (np.shape(Ys2)[0],1)
        Ymu = np.dot(Ks.T,np.linalg.solve(self.K,self.Y))
        return  2*np.sqrt(Ys2)
    def predict(self,Z):
        Kss = self.cov.K(self.hyp,Z,Z)
        Ks = self.cov.K(self.hyp,self.X,Z)
        self.Ymu = np.dot(Ks.T,np.linalg.solve(self.K,self.Y))
        self.pK = Kss - np.dot(Ks.T,np.linalg.solve(self.K,Ks))
        self.Ys2 = np.diag(self.pK)
        self.Ys2.shape = (np.shape(self.Ys2)[0],1)
        self.Z = Z
        return

    def draw(self,Z):
        if self.K == None:
            Ymu = np.zeros(np.shape(Z)[0])
            K = self.cov.K(self.hyp,Z,Z)
            DMu = np.random.multivariate_normal(Ymu,K)
        else:
            self.predict(Z)
            DMu = np.random.multivariate_normal(self.Ymu.flatten(),self.pK)
        
        return(DMu)
    def infer_iter(self,X,Y):
        Knew = self.cov.K(self.hyp,X,X)
        Ks = self.cov.K(self.hyp,self.X,X)
        self.K = np.hstack((np.vstack((self.K,Ks.T)),np.vstack((Ks,Knew))))
        self.Y = np.vstack((self.Y,Y))
        self.X = np.vstack((self.X,X))
        return
    def infer(self,X,Y):
        self.K = self.cov.K(self.hyp,X,X)
        self.Y = Y
        self.X = X
        return
    
    def infer_shit(self,X,Y):

        return

    def optimise_hyper(self):
        hyp = self.hyp.flatten()
        res = opt.minimize(self.nll,hyp)
        self.hyp = res.x
        print(self.hyp)
        self.infer(self.X,self.Y)
        return

    def nll(self,covhyp):
        K = self.cov.K(covhyp,self.X,self.X)
        (_,Klogdet) = np.linalg.slogdet(K)
        marg = (Klogdet + np.dot(self.Y.T,np.linalg.solve(K ,self.Y)) + np.shape(self.Y)[0] * np.log(2*np.pi))/2  + np.dot(np.exp(covhyp).T,np.exp(covhyp))
        return marg.flatten()


if __name__=="__main__":
    from GPUtility import plot_env, plot_env3d
    x = np.linspace(0,10,num=100)
    (x1,x2) = np.meshgrid(x,x)
    x1.shape=(10000,1)
    x2.shape=(10000,1)
    X = np.hstack((x1,x2));
    y1 = np.cumsum(np.random.randn(100,1))
    y2 = np.cumsum(np.random.randn(100,1))
    (Y1,Y2) = np.meshgrid(y1,y2)
    Y1.shape = (10000,1)
    Y2.shape = (10000,1)
    Y = Y1*Y2
    iarray = np.random.permutation(np.arange(X.shape[0]))[:100]
    XSamp = X[iarray,:]
    YSamp = Y[iarray]
    cov = Plus(MahalanobisDist(Matern5()),Noise())
    hyp = np.random.randn(cov.hyp(2))
    GP = GaussianProcess(hyp,cov)
    
    GP.infer(XSamp, YSamp)
    GP.optimise_hyper()
    GP.predict(X)
    plot_env3d(GP)
    print("Done")
