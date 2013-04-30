from GP import GaussianProcess
from cov.dist.Mahalanobis import MahalanobisDist
from cov.dist.Euclidean import EuclideanDist
from cov.SquaredExponential import SqExp
from cov.Noise import Noise
from cov.Matern import Matern3
from cov.Linear import Linear 
from cov.meta.Plus import Plus
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


cov = Plus(EuclideanDist(Matern3()),Linear())
hyp = np.array([[1],[1],[np.log(0.1)]])
gp = GaussianProcess(hyp,cov)
t = np.linspace(0,100,100)
t.shape = (np.shape(t)[0],1)
gp.infer(np.array([[0]]),np.array([[0]]))
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in xrange(0,10):
    x = gp.draw(t)
    y = gp.draw(t)
    ax.plot(x,y,zs=t)
plt.show()