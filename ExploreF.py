from PolyGaussian import PolyGaussian
import numpy as np
from matplotlib import pyplot as plt
import time
x = np.linspace(-100,100,1000)
x.shape = (1000,1)
l = np.array([0.,0.])
s02 = 0
st2 = 0.01
timestep = 1
tend = 150
maxspeed = 0.1
f = np.abs(np.cumsum(np.random.randn(np.size(x))))
f.shape = (1000,1)
done = False
pg = PolyGaussian(timestep,s02,st2)
mf = np.zeros(np.shape(f))
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
p = pg.eval(x, l[0], l[1], tend)
g = p*f
line1, = ax1.plot(x,f)
line2, = ax2.plot(x,p)
blob1, = ax2.plot(l[0],0,'o',markersize=20)
line3, = ax3.plot(x,mf)
line4, = ax4.plot(x,g)

while not done:
    p = pg.eval(x, l[0], l[1], tend)
    g = p*f
    #Work out if there's greater mass to the left or right of us
    pivot = (np.abs(x-l[0])).argmin()
    mf[pivot] = f[pivot]
    f[pivot] = 0
    left = np.sum(g[:pivot])
    right = np.sum(g[pivot+1:])
    if right > left:
        if l[1] < maxspeed:
            l[1] += st2*timestep
    elif left > right:
        if l[1] > -maxspeed:
            l[1] += -st2*timestep
    else:
        done = True
    
    l[0] += l[1]*timestep
    line1.set_ydata(f)
    line2.set_ydata(p)
    blob1.set_xdata(l[0])
    line3.set_ydata(mf)
    line4.set_ydata(g)
    fig.canvas.draw()
    print(l)
    
