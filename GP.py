import numpy as np
import scipy as sp
from GPUtility import *
import scipy.linalg as LA
import scipy.optimize as opt
from matplotlib import pyplot as plt
class GaussianProcess():
    def __init__(self,hyperparameters=None,mean='zero',kernel='SqExp'):
        self.kernel = kernel
        self.mean = mean
        self.hyperparameters = hyperparameters
        self.sigma = 0 #Assume zero noise obs at first
        self.X = None
        self.y = None
        self.f = None
        self.C = None
        self.A = None
        self.B = None
        self.K = None
        
# Draw a sample from the GP. If we've trained the GP with training points, it draws from the Normal
# conditioned on the training points. Covariance matrices are in this order: [[A, B], [B', C]].
# K is the training kernel, Ks is the gramm of training vs test and Kss is the test covariance
    def draw(self,x):
        self.condition(x)
        return np.random.multivariate_normal(self.mean,self.C)

    def condition(self,x):
        Kss = gramm(x,x,self.hyperparameters)
        if self.K == None:
            self.mean = np.zeros(np.size(x))
            self.C = Kss
        else:
            Ks = gramm(self.X,x,self.hyperparameters)
            self.mean = Ks.T.dot(np.linalg.inv(self.K).dot(self.f.T))
            self.C = Kss - Ks.T.dot(np.linalg.inv(self.K).dot(Ks))
        return
    
    def learn(self,X,f,sigma=0):
        KNew = gramm(X,X,self.hyperparameters)
        KNew = KNew + sigma*np.eye(np.shape(KNew)[0])
        if self.K == None:
            self.K = KNew
            self.f = f
            self.X = X
        else:
            KSide = gramm(self.X,X,self.hyperparameters)
            self.f = np.hstack((self.f, f))
            self.X = np.hstack((self.X, X))
            self.K = np.vstack((np.hstack((self.K,KSide)),np.hstack((KSide.T,KNew))))
        return

    def optimise_hyper(self):
        #self.hyperparameters = opt.fmin_cg(self.nll,self.hyperparameters)
        self.hyperparameters = opt.brute(self.nll,[(-1,1),(-1,1),(-1,1),(-1,1)])
        return self.hyperparameters

#Calculates the log-likelihood of the GP for its given hyperparameters... I think
    def loglikelihood(self,theta):
        K = gramm(self.X,self.X,theta) + np.eye(np.size(self.X))*0.001
        sign,Klogdet = np.linalg.slogdet(K)
        marginal = -0.5 * np.matrix(self.f) * np.linalg.inv(np.matrix(K)) * np.matrix(self.f.T) - 0.5 * Klogdet - (self.f.size/2.) * np.log(2*np.pi)
        return np.array(marginal).flatten()

    def nll(self,theta):
        return -self.loglikelihood(theta)

    
#    def condition(self,X
#When x and y are jsintly gaussian, this returns conditional of y given x:
#    def condition_gaussian(self):
#        self.C = B - C * np.linalg.inv(A) * C
#        self.fstar = mu2 + C * np.linalg.inv(B) * f
        
        
