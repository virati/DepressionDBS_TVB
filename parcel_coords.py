#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:22:30 2018

@author: virati
Script to extract parcellation information/centers and dump to external log file
"""
import sys
sys.path.append('/home/virati/Dropbox/projects/libs/tvb-library/')
from tvb.simulator.lab import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

import scipy.signal as sig

#%%

conn = connectivity.Connectivity.from_file("connectivity_192.zip")

center_coords = conn.centres
# write it out?
np.save('/home/virati/Dropbox/TVB_192_coord.npy',center_coords)
np.save('/home/virati/Dropbox/TVB_192_conn.npy',conn.weights)