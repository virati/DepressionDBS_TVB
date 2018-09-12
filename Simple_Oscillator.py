# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tvb
from tvb.simulator.lab import *
#from tvb.simulator.lab import models, connectivity, integrators, monitors, coupling, simulator


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx



import mne

#%%

#oscillator = models.Generic2dOscillator()
jrm = models.JansenRit(mu=0.0,v0=6.0)

#%%
# Handle Connectivity now
#white_matter = connectivity.Connectivity(load_default=True)
white_matter = connectivity.Connectivity.from_file("connectivity_192.zip")
white_matter.speed = np.array([4.0])
white_matter_coupling = coupling.SigmoidalJansenRit(a=10)


if 0:
    white_matter.configure()
    plot_connectivity(connectivity=white_matter)


#%%
phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max - jrm.p_min) * 0.5)**2 / 2

sigma = np.zeros(6)
sigma[3] = phi_n_scaling

heunint = integrators.HeunStochastic(dt=2**-4,noise=noise.Additive(nsig=sigma))

mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_raw,mon_tavg)

#%%
sens_eeg = sensors.SensorsEEG(load_default=True)
skin = surfaces.SkullSkin(load_default=True)
skin.configure()

sens_eeg.configure()
if 0:
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    x,y,z = white_matter.centres.T
    ax.plot(x,y,z,'ko')
    
    sx,sy,sz = sens_eeg.sensors_to_surface(skin).T

#%%
# Stimulation here
eqn_t = equations.PulseTrain()
eqn_t.parameters['onset']=1.5e2
eqn_t.parameters['T'] = 100.0
eqn_t.parameters['tau'] = 50.0

weights = np.zeros((192,))
weights[[14,52,100,120]]=0.9

stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=white_matter,weight=weights)

    
#%%

sim = simulator.Simulator(model=jrm, connectivity=white_matter,coupling=white_matter_coupling,integrator=heunint,monitors=what_to_watch,simulation_length=1e3,stimulus=stimulus)
sim.configure()

(time,data) = sim.run()

voltages = data[1]

#%%
plt.figure()
plt.plot(time[0][:4000],voltages[:,0,:,0].squeeze(),'k',alpha=0.1)



#%%
def DEPR_plot_time():
    
    ###
    raw_data = []
    raw_time = []
    tavg_data = []
    tavg_time = []
    
    for raw,tavg in sim(simulation_length=2**10):
        
        
        if not raw is None:
            raw_time.append(raw[0])
            raw_data.append(raw[1])
            
        if not tavg is None:
            tavg_time.append(tavg[0])
            tavg_data.append(tavg[1])
            
            
    RAW = np.array(raw_data)
    TAVG = np.array(tavg_data)
    
    #%%
    plt.figure(1)
    plt.plot(raw_time,RAW[:,0,:,0])
    plt.title('Raw - State Var 0')
    
    plt.figure(2)
    plt.plot(tavg_time,TAVG[:,0,:,0])
    plt.title('Temporal Avg')
    
    plt.show()



#%%
# NetworkX stuff

def nx_conn(white_matter):
    G = nx.from_numpy_matrix(white_matter.weights)
    
    plt.figure()
    nx.draw_random(G)
    plt.xlim((-0.2,0.2))
