#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:54:38 2018

@author: virati
Resting Network Tutorial and extensions
"""

import sys
from tvb.simulator.lab import *
import scipy.stats
from sklearn.decomposition import FastICA
import time

#%%



def run_sim(conn,cs, D, cv=3.0, dt=0.5, simlen=1e3):
    sim = simulator.Simulator(
            model=models.Generic2dOscillator(a=0.0),
            connectivity=conn,
            coupling=coupling.Linear(a=cs),
            integrator=integrators.HeunStochastic(dt=dt,noise=noise.Additive(nsig=array([D]))),
            monitors=monitors.TemporalAverage(period=5.0)
            )
    
    sim.configure()
    
    (t,y), = sim.run(simulation_length=simlen)
    return t, y[:,0,:,0]


conn = connectivity.Connectivity(load_default=True)