from PolyGaussian import PolyGaussian
import numpy as np
from matplotlib import pyplot as plt
from Agent import Agent
import time
import random
from cov.dist.Mahalanobis import MahalanobisDist
from cov.Matern import Matern3
from cov.SquaredExponential import SqExp
from cov.meta.Plus import Plus
from cov.Noise import Noise
from cov.dist.Euclidean import EuclideanDist
from GP import GaussianProcess

def auction(agents,x,f,tend):
    localAgents = list(agents)
    random.shuffle(localAgents)
    infoLoss = np.zeros(np.shape(f))
    rnd = 1
    while localAgents:
        rnd += 1
        gain = [0] * len(localAgents)
        info = [0] * len(localAgents) 
        for i,agent in enumerate(localAgents): #Make each remaining agent bid
            (gain[i],info[i]) = agent.propose(x,f,tend,infoLoss)
        iBest = gain.index(max(gain))
        del(localAgents[iBest]) #Remove the winning agent from the bid pool
        infoLoss += info[iBest]
def draw_world(gp,x,f,agents,lines2,lines,blobs,ax1,ax2):
    gp.predict(x)
    colors = ['r','g','b']
    ax1.clear()   
    for i in xrange(0,len(agents)):
        blobs[i], = ax1.plot(agents[i].x,0,'o',markersize=20,color=colors[i])

        
     
    
    ax1.plot(x,gp.Ymu,color='k')
    ax1.plot(x,f,color='m')
    #ax1.fill_between(x.flatten(),np.zeros((np.shape(f))).flatten(),np.zeros((np.shape(f))).flatten(),alpha=0.5,color='grey')
    #lines[4].set_ydata(gp.Ymu)
    ax1.fill_between(x.flatten(),(gp.Ymu-gp.Ys2*2).flatten(),(gp.Ymu + gp.Ys2*2).flatten(),alpha=0.5,color='grey')
    #lines[5] = ax1.lines[5]
    
    #for i in xrange(0,3):
    #    blobs[i].set_xdata(agents[i].x)
    ax2.clear()
    ax2.plot(x,gp.Ys2)
    
def run_sim():        
    wsize = 1000
    x = np.linspace(-100,100,wsize)
    x.shape = (wsize,1)
    f = np.abs(np.cumsum(np.random.randn(np.size(x))))
    f.shape = (wsize,1)
    f = (f - np.mean(f))/np.mean(f)
    of = np.copy(f)
    af = f[:]
    done = False
    sampled = [False for i in xrange(0,wsize)]
    cov = Plus(EuclideanDist(Matern3()),Noise())
    hyp = np.array([[np.log(1)],[np.log(1)],[np.log(10)]])
    gp = GaussianProcess(hyp,cov)
    gpX = np.array([[]])
    gpY = np.array([[]])
    
    vmax = 1.1
    s02 = 0
    st2 = 0.1
    tend = 100
    v0 = 0
    x0 = 0
    dt = 1
    nagent = 2
    
    plt.ion()
    lines = [0] * (nagent + 3)
    blobs = [0] * nagent
    agents = [0] * nagent
    lines2 = [0] * nagent
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    colors = ['r','g','b']
    
    xA = x0 + np.random.randn()*50
    vmaxA = vmax * np.abs(np.random.randn())
    agents[0] = Agent(-0,0,dt,0.11,0.11,1.1001,1)
    agents[1] = Agent(0,0,dt,0.11,0.11,0.303,2)
    for i in xrange(0,nagent):
        blobs[i], = ax1.plot(agents[i].x,0,'o',markersize=20,color=colors[i])
        lines[i], = ax1.plot(x,np.zeros(np.shape(x)),color=colors[i])
        
        
    lines[nagent], = ax1.plot(x,f,color='m')
    lines[nagent+1], = ax1.plot(x,np.zeros((np.shape(f))),color='k')
    lines[nagent+2] = ax1.fill_between(x.flatten(),np.zeros((np.shape(f))).flatten(),np.zeros((np.shape(f))).flatten(),alpha=0.5,color='grey')
    firstRun = True
    while not done:
        if firstRun:
            gp.Ys2 = np.ones(np.shape(x))
        else:
            gp.predict(x)
        auction(agents,x,gp.Ys2,tend)
        #Agents act.
        for agent in agents:
            agent.act(dt)
            idx = agent.find_idx(x)
            gpX = np.append(gpX,x[idx])
            gpY = np.append(gpY,f[idx])
            gpX.shape = (gpX.shape[0],1)
            gpY.shape = (gpY.shape[0],1)
            gp.infer(gpX, gpY)   
            gp.optimise_hyper_brute();
        draw_world(gp,x,of,agents,lines2,lines,blobs,ax1,ax2)      
        done = all(sampled)
        fig.canvas.draw()
if __name__=="__main__":
    run_sim()