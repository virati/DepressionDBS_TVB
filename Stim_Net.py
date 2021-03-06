#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:54:38 2018

@author: virati
Main file for stimulation network with OnT/OffT 130Hz tractography informed stimulation
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/libs/tvb-library/')
from tvb.simulator.lab import *
import scipy.stats
from sklearn.decomposition import FastICA
import time
import numpy as np
import matplotlib.pyplot as plt

#%%
conn = connectivity.Connectivity(load_default=True)

#conn = connectivity.Connectivity.from_file("connectivity_192.zip")
weighting = np.zeros((76,))
weighting[[14, 52, 11, 49]] = 100


eqn_t = equations.PulseTrain()
eqn_t.parameters['onset'] = 2e3
eqn_t.parameters['T'] = 100.0
eqn_t.parameters['tau'] = 50.0

stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=conn,weight=weighting)

stimulus.configure_space()
stimulus.configure_time(np.arange(0.,3e3,2**-4))

plot_pattern(stimulus)

#%%
JRmodel = models.JansenRit(mu=0.0,v0=6.0,r=0.3,J=200)
coupling = coupling.SigmoidalJansenRit(a=10.0)


#G2model = models.Generic2dOscillator(a=0.3,tau=2)
#coupling = coupling.Difference(a=7e-4)

sim = simulator.Simulator(model=JRmodel,
                          connectivity=conn,
                          coupling=coupling,
                          integrator=integrators.HeunStochastic(dt=0.5,noise=noise.Additive(nsig=5e-5)),
                          monitors=(monitors.TemporalAverage(period=1.0),),
                          stimulus=stimulus,
                          simulation_length=5e3,).configure()

(tavg_time,tavg_data), = sim.run()

#%%
plt.figure()
plt.plot(tavg_time, tavg_data[:, 0, :, 0], 'k', alpha=0.1)
plt.plot(tavg_time, tavg_data[:, 0, :, 0].mean(axis=1), 'r', alpha=1)
plt.ylabel("Temporal average")
plt.xlabel('Time (ms)')
