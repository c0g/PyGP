import numpy as np

def ltri(v):
    N = np.shape(v)[0]
    dim = int(0.5*(-1 + np.sqrt(8*N + 1)))
    V = np.zeros((dim,dim))
    s=0
    for o in xrange(dim,0,-1):
        k = dim - o
        V = V + np.diagflat(v[s:s+o],-k);
        s = s+o
    return V

if __name__ == "__main__":
    v = np.linspace(2,5,num=6)
    print(v)
    A = ltri(v)
    print(A)
