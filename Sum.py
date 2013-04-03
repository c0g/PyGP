import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
n=500
t = np.linspace(1,10,n)
x = np.linspace(-5,20,500)
p = np.zeros((np.shape(x)))
s2 = 0.5
xdot = -0.5
for i in xrange(int(n)):
    s2i = (s2**2)*t[i]
    mu = t[i]*xdot
    p += stats.norm.pdf(x,loc=mu,scale=s2i)
    
p = p/n
plt.plot(x,p)
plt.show()