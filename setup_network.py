#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:16:20 2018

@author: virati
"""

import networkx as nx
import numpy as np
import scipy.signal as sig

#from tvb.simulator.lab import *

class DN:
    state = np.array([])
    def __init__(self,statedim=2):
        self.G = nx.Graph()
        self.state = np.zeros((statedim,1))
        
    def tstep(self):
        pass
    
    def dynamics(self):
        x = {idx:val for }
    
    def plot_dyn(self):
        pass
    
    def lie_brack(self):
        pass
    
    def set_h(self):
        pass
    
    
mainNet = DN()
mainNet.plot_dyn()