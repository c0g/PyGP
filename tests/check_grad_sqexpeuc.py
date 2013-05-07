import numpy as np
import scipy.optimize as opt
from cov.SquaredExponentialEuc import SqExpEuc
hyp = np.array([0.1,0.1,0.1])
Z1 = Z2 = np.linspace(0,10,50)
Z1.shape = Z2.shape = (50,1)
obs = np.sin(Z1) + np.random.randn(50,1)*01
cov = SqExpEuc()


def loglik(hyp,cov,Z1,Z2,obs,prior):
    lik = np.exp(hyp.flatten()[0]*2)
    likMat = np.eye(np.shape(Z1)[0]) * lik
    hyp2 = hyp.flatten()[1:]
    K = cov.K(hyp2,Z1,Z2)[0]
    C = K + likMat
    L = np.linalg.cholesky(C)
    Cnt = np.linalg.solve(L.T,np.linalg.solve(L,obs))
    (s,CLogDet) = np.linalg.slogdet(C)
    ll = -0.5 * CLogDet - 0.5 * obs.T.dot(Cnt) - np.size(obs)*np.log(2*np.pi)/2 + np.exp(hyp*2).flatten().T.dot(prior.dot(np.exp(hyp*2).flatten()))
    return ll

def dloglik(hyp,cov,Z1,Z2,obs,prior):
    lik = np.exp(hyp.flatten()[0]*2)
    likMat = np.eye(np.shape(Z1)[0]) * lik
    hyp2 = hyp.flatten()[1:]
    kDk= cov.K(hyp2,Z1,Z2)
    K = kDk[0]
    C = K + likMat
    L = np.linalg.cholesky(C)
    dk = kDk[1]
    Cnt = np.linalg.solve(L.T,np.linalg.solve(L,obs))
    dK = (2*likMat,dk[0],dk[1])
    
    drivs = [np.array([])]*(len(hyp))

    for i,d in enumerate(dK):
        CnDCn = np.linalg.solve(L.T,np.linalg.solve(L,d))
        drivs[i] = (-np.trace(CnDCn) + obs.T.dot(CnDCn).dot(Cnt))/2
    return np.array(drivs).flatten() +  4*np.exp(hyp.flatten()*4).T.dot(prior)


covF = lambda hyp,z1,z2: cov.K(hyp,z1,z2)[0]
covdF = lambda hyp,z1,z2: np.array(cov.K(hyp,z1,z2)[1]).flatten()
hyp = np.array([1,1])
print("Cov deriv check: " + str(opt.check_grad(covF,covdF,hyp,np.array([[100,0,0]]),np.array([[100,0,0]]))))
hyp = np.array([0.1,1,1])
prior = np.array([[1,0,0],[0,1,0],[0,0,1]])
print("LogLik deriv check: " +str(opt.check_grad(loglik,dloglik,hyp,cov,Z1,Z2,obs,prior)))
#print(opt.minimize(loglik,hyp,args=(cov,Z1,Z2,obs,prior)))
ret=opt.minimize(loglik,hyp,jac=dloglik,method='L-BFGS-B',args=(cov,Z1,Z2,obs,prior))
print(ret)
print(ret.x)
