#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 23:04:19 2018

@author: virati
This file essentially replaces the version I made a while back. Can mess around with phase portraits of 2D oscillatory
"""

from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

oscillator = models.Generic2dOscillator()
#oscillator = models.HopfNet()
#oscillator = models.Kuramoto()

ppi_fig = PhasePlaneInteractive(model=oscillator)
ppi_fig.show()