from GP import GaussianProcess
from cov.dist.Mahalanobis import MahalanobisDist
from cov.SquaredExponential import SqExp
from cov.Noise import Noise
from cov.Matern import Matern3
from matplotlib import pyplot as plt
import numpy as np

def draw_path():
    cov = MahalanobisDist(SqExp())
    nHyp = cov.hyp(2)
    hyp = np.array([[0.1],[100]])
    GP = GaussianProcess(hyp,cov)
    tEnd = 60.
    n = 1000
    T = np.linspace(0,tEnd,n)
    dT = tEnd/n
    T.shape = (np.shape(T)[0],1)
    plt.hold(True)
    for i in xrange(50):
        print(i)
        AX = GP.draw(T)
        AY = GP.draw(T)
        VX = np.cumsum(AX)*dT
        VY = np.cumsum(AY)*dT
        X = np.cumsum(VX)*dT
        Y = np.cumsum(VY)*dT
        plt.plot(X,Y)
    
    plt.show()
    

if __name__=="__main__":
    draw_path()