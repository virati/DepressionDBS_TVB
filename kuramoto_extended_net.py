#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:51:59 2018

@author: virati
Kuramoto Network
EXTENDED where each node x_i is a vector in \mathbf{R}^k that can be parameterized to \mathbf{R}
"""

import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig




phases = np.random.random()
