#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:15:40 2018

@author: virati
Modeled SCCwm-DBS
"""
from tvb.simulator.lab import *
import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as sig

#%%

conn = connectivity.Connectivity.from_file("connectivity_192.zip")



#%% This is stimulus information
# configure stimulus spatial pattern
weighting = np.zeros((192, ))
g_locs = [40]
weighting[g_locs] = 0.1

eqn_t = equations.PulseTrain()
eqn_t.parameters['onset'] = 1.5e3
eqn_t.parameters['T'] = 100.0
eqn_t.parameters['tau'] = 50.0

stimulus = patterns.StimuliRegion(
    temporal=eqn_t,
    connectivity=conn,
    weight=weighting)

#Configure space and time
stimulus.configure_space()
stimulus.configure_time(np.arange(0., 3e3, 2**-4))

#And take a look
plot_pattern(stimulus)
#%%
coupling_style = coupling.Difference(a=7e-3)

#%%
#Run our simulator
sim = simulator.Simulator(
    model=models.Generic2dOscillator(a=0.4, tau=2), #typically a=0.3,tau=2
    connectivity=conn,
    coupling=coupling_style,
    integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=5e-5)),
    monitors=(
        monitors.TemporalAverage(period=1.0),
        ),
    stimulus=stimulus,
    simulation_length=10e3, # 1 minute simulation
).configure()

(tavg_time, tavg_data),  = sim.run()


#%%
plt.figure()
plt.plot(tavg_time, tavg_data[:, 0, :, 0], 'k', alpha=0.1)
plt.plot(tavg_time, tavg_data[:, 0, :, 0].mean(axis=1), 'r', alpha=1)
plt.plot(tavg_time,tavg_data[:,0,g_locs,0],'b',linewidth=2)
plt.ylabel("Temporal average")
plt.xlabel('Time (ms)')


#%%
#find the neighbors of a given node
lapl = conn.weights
node_oi = 40
secondary_nodes = np.where(lapl[node_oi,:] > 0)[0]


#%%
#Do some TF plotting
for plot_node in [40]:#secondary_nodes:
    print(plot_node)
    tser = sig.decimate(tavg_data[:,0,plot_node,0],2)
    F,T,SG = sig.spectrogram(tser,fs=1000/2,window=sig.blackmanharris(512),nfft=512,noverlap=500)
    plt.figure()
    plt.pcolormesh(T,F,np.log10(SG))