from PolyGaussian2d import PolyGaussian
import numpy as np
from matplotlib import pyplot as plt
from Agent2d import Agent
import time

def auction(agents,X,Y,f,tend):
    localAgents = list(agents)
    infoLoss = np.zeros(np.shape(f))
    rnd = 1
    while localAgents:
        print("\nRound " + str(rnd))
        rnd += 1
        gain = [0] * len(localAgents)
        info = [0] * len(localAgents) 
        for i,agent in enumerate(localAgents): #Make each remaining agent bid
            (gain[i],info[i]) = agent.propose(X,Y,f,tend,infoLoss)
        iBest = gain.index(max(gain))
        print("\nWon by :" + str(localAgents[iBest]) + "\nWith gain: " + str(gain[iBest]))
        del(localAgents[iBest]) #Remove the winning agent from the bid pool
        infoLoss += info[iBest]
        
def draw_world(x,agents,blobs):
    for i,agent in enumerate(agents):
        blobs[i].set_xdata(np.append(blobs[i].get_xdata(),agents[i].x))
        blobs[i].set_xdata(np.append(blobs[i].get_xdata(),agents[i].x))
        
    
def run_sim():        
    wsize = 100
    x = np.linspace(-100,100,wsize)
    y = np.linspace(-100,100,wsize)
    x.shape = (wsize,1)
    y.shape = (wsize,1)
    (X,Y) = np.meshgrid(x, y)
    f = np.zeros((wsize,wsize))
    for i in xrange(0,3):
        L = np.random.randn(2,2)
        C = np.dot(L.T,L)
        xsig = C[0,0]*20+1
        ysig = C[1,1]*20+1
        xysig = C[1,0]*20+1
        xmu = 20*np.random.randn(1)
        ymu = 20*np.random.randn(1)
        f += np.exp(-0.5*(((X-xmu)/xsig)**2 + ((Y-ymu)/ysig)**2 + 2*(X-xmu)*(Y-ymu)/(xysig**2)))
    f /= 3    
    f.shape = (wsize,wsize)
    
    vmax = 0.1
    s02 = 0
    st2 = 0.01
    tend = 1000
    v0 = np.array([[0],[0]])
    x0 = np.array([[0],[0]])
    
    dt = 1
    nagent = 3
    
    plt.ion()
    lines = [0] * (nagent + 1)
    blobs = [0] * nagent
    agents = [0] * nagent
    lines2 = [0] * nagent
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in xrange(0,nagent):
        xA = x0 + np.random.randn(2,1)*20
        agents[i] = Agent(xA,v0,dt,s02,st2,vmax,i)
        blobs[i], = ax1.plot(agents[i].x[0],agents[i].x[1],'o',markersize=5)
        
    x.shape = (wsize,)
    y.shape = (wsize,)    
    ax1.contour(x,y,f)
    x.shape = (wsize,1)
    y.shape = (wsize,1) 
    done = False
    while not done:
        auction(agents,X,Y,f,tend)
        #Agents act.
        for agent in agents:
            agent.act(dt)
            idx = agent.find_idx(x)
            if not sampled[idx]:
                af[idx] = f[idx]
                f[idx] = 0
                sampled[idx] |= True
        draw_world(x,f,agents,lines2,lines,blobs)      
        done = all(sampled)
        fig.canvas.draw()
if __name__=="__main__":
    import cProfile
    #cProfile.run('run_sim()')
    run_sim()