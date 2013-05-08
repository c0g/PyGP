import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from itertools import izip

def plot_env(GP):

    sigma = np.sqrt(GP.Ys2)
    top = GP.Ymu + 2*sigma
    bottom = GP.Ymu - 2*sigma
    plt.hold(True)
    plt.plot(GP.X,GP.Y,marker='o',linestyle='')
    plt.plot(GP.X,GP.T,linestyle='-')
    plt.plot(GP.Z,GP.Ymu,linestyle='--')
    plt.fill_between(GP.Z.flatten(),top.flatten(),bottom.flatten(),alpha=0.5,color='grey')
    plt.show()
    
def plot_env3d(GP):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.axes3d import get_test_data
    fig = plt.figure()
    X = GP.Z[:,0]
    X.shape = (100,100)
    Y = GP.Z[:,1]
    Y.shape = (100,100)
    Ys = np.sqrt(GP.Ys2)
    Ys.shape = (100,100)
    ax = fig.add_subplot(211, projection='3d')
    Z = GP.Y
    Z.shape = (100,100)
    plt.hold(True)
    ax.plot_surface(X,Y,Z)
    ax = fig.add_subplot(212, projection='3d')
    Z = GP.Ymu
    Z.shape(100,100)
    ax.plot_surface(X,Y,Z)
    plt.show()
    

class Hyperparameters:
    def __init__(self,mn,cov,lik):
        self.mn = mn
        self.cov = cov
        self.lik = lik
        
        


