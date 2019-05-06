# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/libs/tvb-library/')
import tvb
from tvb.simulator.lab import *
#from tvb.simulator.lab import models, connectivity, integrators, monitors, coupling, simulator


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy
import numpy.random as random

import scipy.signal as sig

#import mne

#%%
oscillator = models.Generic2dOscillator(a=2.0,b=-10.0,c=0.0,d=0.02,I=0.0)
jrm = models.JansenRit(mu=0.0,v0=6.0)
kuramoto = models.Kuramoto(omega=10)

use_model = jrm

#%%
# Handle Connectivity now
#white_matter = connectivity.Connectivity(load_default=True)
white_matter = connectivity.Connectivity.from_file("connectivity_192.zip")
white_matter.speed = np.array([4.0])
white_matter_coupling = coupling.Difference(a=1)
#white_matter_coupling = coupling.Linear()

#white_matter_coupling = coupling.SigmoidalJansenRit(a=10)


def plot_conn():
    white_matter.configure()
    plot_connectivity(connectivity=white_matter)
#plot_conn()

#%%
phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max - jrm.p_min) * 0.5)**2 / 2

sigma = np.zeros((6,192))
sigma[3] = phi_n_scaling

heunint = integrators.HeunStochastic(dt=2**-4,noise=noise.Additive(nsig=sigma))


#%%
#Monitors/things to watch

mon_raw = monitors.Raw()
#This temporally averages??
mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_raw,mon_tavg)
#what_to_watch = (mon_raw)

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
eqn_t.parameters['onset']=1.5e3
eqn_t.parameters['T'] = 100.0
eqn_t.parameters['tau'] = 50.0

weights = np.zeros((2,192))
#make a random set of N values between 0 and 192
stim_nodes = np.ceil(191*random.sample(20)).astype(int)
#stim_nodes = [100]
weights[:,stim_nodes]=40

stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=white_matter,weight=weights)
stimulus.configure_space()
stimulus.configure_time(numpy.arange(0,3e3,2**-4))
#plot_pattern(stimulus)
    
#%%
# Main simulator configuration
sim = simulator.Simulator(model=use_model, connectivity=white_matter,coupling=white_matter_coupling,integrator=heunint,monitors=what_to_watch,simulation_length=5e3,stimulus=stimulus)
sim.configure()

#%%
# Main simulation run
(time,data) = sim.run()

voltages = data[1]

#%%
plt.figure()
for ii in range(4):
    plt.subplot(4,1,ii+1)
    plt.plot(voltages[:,ii,:,0].squeeze(),'k',alpha=0.1)
    
    
#%%
select_state = 2
NFFT = 512
plt.figure()
F,T,SG = sig.spectrogram(voltages[:,select_state,2,0].squeeze().T,nperseg=NFFT,noverlap=0.5*NFFT,window=sig.get_window('blackmanharris',NFFT),fs=16)

plt.pcolormesh(T,F,10*np.log10(SG),rasterized=True)

#%%
# NetworkX stuff

def nx_conn(white_matter):
    G = nx.from_numpy_matrix(white_matter.weights)
    
    plt.figure()
    nx.draw_random(G)
    plt.xlim((-0.2,0.2))
