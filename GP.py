import numpy as np
from scipy import optimize as opt

#OBSERVATIONS ARE STORED IN NxD MATRICES. N rows, D columns. Hence, obs are //horizontal//
class GaussianProcess():
    def __init__(self,hyp,cov):
        self.cov = cov
        self.hyp = hyp
        self.Ymu =np.array([]) 
        self.Ys2 = np.array([])
        self.Z = np.array([])
        self.F = np.array([])
        self.Zp = np.array([])
        self.T = np.array([])
        self.K = np.array([])
        
    def predict(self,Zp):
        Kss = self.cov.K(self.hyp,Zp,Zp)
        Ks = self.cov.K(self.hyp,self.Z,Zp)
        self.Ymu = np.dot(Ks.T,np.linalg.solve(self.K,self.F))
        self.pK = Kss - np.dot(Ks.T,np.linalg.solve(self.K,Ks))
        self.Ys2 = np.diag(self.pK)
        self.Ys2.shape = (np.shape(self.Ys2)[0],1)
        self.Zp = Zp
        return

    def infer(self,Z,Fobs):
        Knew = self.cov.K(self.hyp,Z,Z)
        if self.K.size != 0:
            Ks = self.cov.K(self.hyp,self.Z,Z)
            self.K = np.hstack((np.vstack((self.K,Ks.T)),np.vstack((Ks,Knew))))
            self.F = np.vstack((self.F,Fobs))
            self.Z = np.vstack((self.Z,Z))
        else:
            self.K = Knew
            self.F = np.array([Fobs])
            self.Z = Z
        return
    
    def optimise_hyper(self):
        hyp = self.hyp.flatten()
        res = opt.minimize(self.nll,hyp)
        self.hyp = res.x
        self.infer(self.X,self.Y)
        return

    def nll(self,covhyp):
        K = self.cov.K(covhyp,self.X,self.X)
        (_,Klogdet) = np.linalg.slogdet(K)
        marg = (Klogdet + np.dot(self.Y.T,np.linalg.solve(K ,self.Y)) + 
                np.shape(self.Y)[0] * np.log(2*np.pi))/2  + np.dot(np.exp(covhyp).T,np.exp(covhyp))
        return marg.flatten()
