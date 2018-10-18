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
import numpy
import numpy.random as random


#import mne

#%%

#oscillator = models.Generic2dOscillator()
jrm = models.JansenRit(mu=0.0,v0=6.0)

#%%
# Handle Connectivity now
#white_matter = connectivity.Connectivity(load_default=True)
white_matter = connectivity.Connectivity.from_file("connectivity_192.zip")
white_matter.speed = np.array([4.0])
white_matter_coupling = coupling.SigmoidalJansenRit(a=10)


def plot_conn():
    white_matter.configure()
    plot_connectivity(connectivity=white_matter)


#%%
phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max - jrm.p_min) * 0.5)**2 / 2

sigma = np.zeros(6)
sigma[3] = phi_n_scaling

heunint = integrators.HeunStochastic(dt=2**-4,noise=noise.Additive(nsig=sigma))


#%%
#Monitors/things to watch

mon_raw = monitors.Raw()
#This temporally averages??
mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_raw,mon_tavg)

#%%
sens_eeg = sensors.SensorsEEG(load_default=True)
skin = surfaces.SkullSkin(load_default=True)
skin.configure()

sens_eeg.configure()

def plot_sensors():
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    x,y,z = white_matter.centres.T
    ax.plot(x,y,z,'ko')
    
    sx,sy,sz = sens_eeg.sensors_to_surface(skin).T

#%%
# Stimulation here
eqn_t = equations.PulseTrain()
eqn_t.parameters['onset']=2.5e3
eqn_t.parameters['T'] = 100.0
eqn_t.parameters['tau'] = 50.0

weights = np.zeros((192,))
#make a random set of N values between 0 and 192
stim_nodes = np.ceil(192*random.sample(20)).astype(int)
weights[stim_nodes]=0.9

stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=white_matter,weight=weights)
stimulus.configure_space()
stimulus.configure_time(numpy.arange(0,3e3,2**-4))
#plot_pattern(stimulus)
    
#%%
# Main simulator configuration
sim = simulator.Simulator(model=jrm, connectivity=white_matter,coupling=white_matter_coupling,integrator=heunint,monitors=what_to_watch,simulation_length=5e3,stimulus=stimulus)
sim.configure()

#%%
# Main simulation run
(time,data) = sim.run()

voltages = data[1]

#%%
plt.figure()
plt.plot(time[0][:20000],voltages[:,0,:,0].squeeze(),'k',alpha=0.1)

#%%
# NetworkX stuff

def nx_conn(white_matter):
    G = nx.from_numpy_matrix(white_matter.weights)
    
    plt.figure()
    nx.draw_random(G)
    plt.xlim((-0.2,0.2))
