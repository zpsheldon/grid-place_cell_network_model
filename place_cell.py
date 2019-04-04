#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:20:57 2019

@author: Zach Sheldon
"""
from scipy.stats import uniform
import numpy as np

class PlaceCell:
    # initializes cell with fields and gaussian parameter sigma
    def __init__(self, x_center, y_center, sigma, x_dim, y_dim):
        self.x_center = x_center
        self.y_center = y_center
        self.sigma = sigma
        self.place_activity = []
        self.spike_positions_x = []
        self.spike_positions_y = []
    
    # getters   
    def get_place_activity(self):
        return self.place_activity # returns activity level of place cell at given timestep in Hz
    def get_spike_positions(self):
        return self.spike_positions_x, self.spike_positions_y # returns the x and y positions at which the cell spikes
    def get_centers(self):
        return self.x_center, self.y_center
    
    # evalutes spiking state based on animal's current position
    def evaluate_spiking(self, x, y):
        rand_val = uniform.rvs() # random float for simulated spiking noise
        curr_place_val = (1/(2*np.pi*self.sigma**2)) * np.exp(-((x - self.x_center)**2 + (y - self.y_center)**2) / (2*self.sigma**2))
        curr_place_val = (curr_place_val / 0.00636371) # normalize to between 0 and 1 - based on sigma=5.0
        curr_place_val = 0.1 * curr_place_val # normalize to between 0 and 0.1 - based on equation from Alex's model
        curr_place_val = curr_place_val * 500 * (1.0/10.0) # normalize to 500 * dt/tau * activ_level - output is Hz
        self.place_activity.append(curr_place_val)
        # check if cell does spike
        if (curr_place_val >= rand_val):
            self.spike_positions_x.append(x)
            self.spike_positions_y.append(y)
            return 1, curr_place_val
        else:
            return 0, curr_place_val
    
    #TODO - should each place cell be normalized by overall max or by its individual max?
    
    