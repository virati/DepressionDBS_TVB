# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tvb
from tvb.simulator.lab import *
import numpy as np

import matplotlib.pyplot as plt

import networkx as nx


#%%

oscillator = models.Generic2dOscillator()
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = np.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0154)

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_raw,mon_tavg)

sim = simulator.Simulator(model=oscillator, connectivity=white_matter,coupling=white_matter_coupling,integrator=heunint,monitors=what_to_watch)

sim.configure()


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
import networkx as nx

G = nx.from_numpy_matrix(white_matter.weights)

plt.figure()
nx.draw_random(G)
plt.xlim((-0.2,0.2))
