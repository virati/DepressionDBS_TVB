#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Kuramoto network implemented in The Virtual Brain (http://thevirtualbrain.org/)

    Copyright (C) 2018  Vineet Tiruvadi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Created on Wed Sep 19 13:51:37 2018

@author: virati
A Kuramoto oscillatory network
SIMPLE version where each x_i is just a single scalar phase
"""


from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt

oscillator = models.Kuramoto(omega=0.01)

white_matter = connectivity.Connectivity.from_file("connectivity_192.zip")
white_matter.speed = np.array([1.0])
white_matter_coupling = coupling.Sigmoidal(a=10)

phi_n_scaling = 1
sigma = np.zeros(192)
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

#stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=white_matter,weight=weights)

    
#%%

sim = simulator.Simulator(model=oscillator, connectivity=white_matter,coupling=white_matter_coupling,integrator=heunint,monitors=what_to_watch,simulation_length=1e3)
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


