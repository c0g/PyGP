import numpy as np
from scipy import optimize as opt

def f(x):
    return x[0]**2 + (x[1]-2)**2 + np.sin(x[2])

hyp = np.array([1,1,0])
res = opt.minimize(f,hyp)
print(res)


