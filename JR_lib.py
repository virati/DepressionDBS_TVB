#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:18:43 2018

@author: Sora, Vineet
JansenRit Model for Depression DBS modeling
"""

class JansenRit:
    def __init__(self):
        y = np.zeros((5,1))
        
        self.coeff_dict = {'a':,'b':,'A':,'B':,'C1':,'C2':,'C3':,'C4':}
    
    def JR_diff(y,coeff_dict):
        
    

    def tstep(self):
        dy = self.JR_diff(self.y,self.coeff_dict)
        self.y += 
