import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def gramm(X_n,X_m,theta):
    #Exponential of quadratic gramm
    on = np.ones(np.shape(X_n))
    om = np.ones(np.shape(X_m))
    m1 = X_n.T.dot(om)
    m2 = on.T.dot(X_m)
    m3 = X_n.T.dot(X_m)
    im = np.eye(np.shape(X_n)[0])
    return(np.pi*(theta[0]**2 * np.exp((-0.5*np.abs(np.sin(np.pi*(m1-m2)))**2)/theta[1]**2) +im * theta[2]**2))

def plot_env(x,mu,cov):
    sigma = np.sqrt(cov)
    top = mu + 2*sigma
    bottom = mu - 2*sigma
    plt.hold(True)
    plt.plot(x,mu)
    plt.fill_between(x,top,bottom,alpha=0.5,color='grey')


