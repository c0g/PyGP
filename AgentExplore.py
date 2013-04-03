from PolyGaussian import PolyGaussian
import numpy as np
from matplotlib import pyplot as plt
from Agent import Agent
import time

def auction(agents,x,f,tend):
    localAgents = list(agents)
    infoLoss = np.zeros(np.shape(f))
    rnd = 1
    while localAgents:
        print("\nRound " + str(rnd))
        rnd += 1
        gain = [0] * len(localAgents)
        info = [0] * len(localAgents) 
        for i,agent in enumerate(localAgents): #Make each remaining agent bid
            (gain[i],info[i]) = agent.propose(x,f,tend,infoLoss)
        iBest = gain.index(max(gain))
        print("\nWon by :" + str(localAgents[iBest]) + "\nWith gain: " + str(gain[iBest]))
        del(localAgents[iBest]) #Remove the winning agent from the bid pool
        infoLoss += info[iBest]
def draw_world(x,f,agents,lines2,lines,blobs):
    lines[3].set_ydata(f)
    for i in xrange(0,3):
        blobs[i].set_xdata(agents[i].x)
        lines[i].set_ydata(max(f)*agents[i].p/max(agents[i].p))
        lines2[i].set_ydata(agents[i].g)
    
def run_sim():        
    wsize = 1000
    x = np.linspace(-100,100,wsize)
    x.shape = (wsize,1)
    f = np.abs(np.cumsum(np.random.randn(np.size(x))))
    f.shape = (wsize,1)
    af = list(f)
    done = False
    sampled = [False for i in xrange(0,wsize)]
    
    vmax = 0.1
    s02 = 0
    st2 = 0.01
    tend = 1000
    v0 = 0
    x0 = 0
    dt = 1
    nagent = 3
    
    plt.ion()
    lines = [0] * (nagent + 1)
    blobs = [0] * nagent
    agents = [0] * nagent
    lines2 = [0] * nagent
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in xrange(0,nagent):
        xA = x0 + np.random.randn()*10
        agents[i] = Agent(xA,v0,dt,s02,st2,vmax,i)
        blobs[i], = ax1.plot(agents[i].x,0,'o',markersize=20)
        lines[i], = ax1.plot(x,np.zeros(np.shape(x)))
        lines2[i], = ax2.plot(x,np.zeros(np.shape(x)))
        
        
    lines[3], = ax1.plot(x,f)
    while not done:
        auction(agents,x,f,tend)
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
    cProfile.run('run_sim()')