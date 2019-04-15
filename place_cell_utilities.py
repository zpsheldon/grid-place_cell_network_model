#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:40:01 2019

@author: Zach Sheldon
"""

import numpy as np
from place_cell import PlaceCell
import matplotlib.pyplot as plt
np.random.seed(1)

# parameters
n_place_cells = 16 # number of place cells
sigma = 5.0 # standard deviation for Gaussian place fields
norm_const = 0.00636371 # normalization constant for place cells with sigma = 5.0
x_dim = 130 # x dimension for environment - cm
y_dim = 130 # y dimension for environment - cm
dist_thresh = 20 # min euclidean distance between place cell centers - cm

# helper function for checking if place centers are too close
def check_centers(p_cells, dist_thresh, rand_x_center, rand_y_center):
    for i in range(0, len(p_cells)):
        # get current place cell
        curr_pcell = p_cells[i]
        curr_x_center, curr_y_center = curr_pcell.get_centers() 
        curr_dist = np.sqrt((rand_x_center - curr_x_center)**2 + (rand_y_center - curr_y_center)**2)
        # if too close, return false
        if curr_dist <= dist_thresh:
            return False
        else:
            # check other place cells
            continue
    # if no other centers are too close, return true
    return True

# helper function for creating place centers
def generate_centers(dist_thresh, p_cells):
    # generate random centers
    rand_x_center = np.random.randint(low=0, high=x_dim)
    rand_y_center = np.random.randint(low=0, high=y_dim)
    
    # check if the random centers are too close to other place cells
    while check_centers(p_cells, dist_thresh, rand_x_center, rand_y_center) == False:
        rand_x_center  = np.random.randint(low=0, high=x_dim)
        rand_y_center = np.random.randint(low=0, high=y_dim)

    return rand_x_center, rand_y_center

# creates place cells
def create_place_cells(n_place_cells, sigma, x_dim, y_dim, dist_thresh, map_b=False):
    print('Generating place cells...')
    p_cells = []
    if map_b: # create double the number of place cells
        n_place_cells = int(2.0*n_place_cells)
    for i in range(0, n_place_cells):
        # randomly create place centers
        rand_x_center, rand_y_center = generate_centers(dist_thresh, p_cells)
        # create place cell and add to list of other cells
        p_cells.append(PlaceCell(rand_x_center, rand_y_center, sigma, x_dim, y_dim))
    return p_cells

# plot gaussian place cell centers
def plot_centers(place_cells, dist_thresh):
    # check if any are within distance threshold
    for i in range(0, len(place_cells)):
        curr_x_center, curr_y_center = place_cells[i].get_centers()
        for j in range(i+1, n_place_cells):
            curr_x_to_compare, curr_y_to_compare = place_cells[j].get_centers()
            curr_dist = np.sqrt((curr_x_to_compare - curr_x_center)**2 + (curr_y_to_compare - curr_y_center)**2)
            if curr_dist <= dist_thresh:
                print('Place centers too close: ', (curr_x_center, curr_y_center), (curr_x_to_compare, curr_y_to_compare), int(curr_dist))

    p_centers = np.zeros((x_dim, y_dim))
    for i in range(0, len(place_cells)):
        curr_x_center, curr_y_center = place_cells[i].get_centers()
        for j in range(0, x_dim):
            for k in range(0, y_dim):
                curr_place_val = (1/(2*np.pi*sigma**2)) * np.exp(-((j - curr_x_center)**2 + (k - curr_y_center)**2) / (2*sigma**2))
                curr_place_val = curr_place_val / norm_const # normalized for sigma = 5.0
                p_centers[k][j] = p_centers[k][j] + curr_place_val
    plt.figure(figsize=(5,5))
    plt.imshow(p_centers, cmap='hot')
    plt.gca().invert_yaxis()

# updates place cell spiking and place activity matrices given spiking at current position x and y at current time index
def evaluate_spiking(place_cell_spiking, place_activity, place_cells, x, y, time_idx):
    for i in range(0, len(place_cells)):
        curr_place_cell = place_cells[i]
        place_cell_spiking[i, time_idx], place_activity[i, time_idx] = curr_place_cell.evaluate_spiking(x[time_idx], y[time_idx])
    return place_cell_spiking, place_activity

# get spiking positions
def get_spiking_positions(place_cells):
    x_spiking_positions = []
    y_spiking_positions = []
    for i in range(0, len(place_cells)):
        curr_place_cell = place_cells[i]
        curr_x_spiking_pos, curr_y_spiking_pos = curr_place_cell.get_spike_positions()
        x_spiking_positions.append(curr_x_spiking_pos)
        y_spiking_positions.append(curr_y_spiking_pos)
    return x_spiking_positions, y_spiking_positions

# plot spiking positions over trajectory
def plot_spike_positions(x, y, x_spiking_positions, y_spiking_positions, n_place_cells):
    print('Plotting spike positions over trajectory...')
    # plot trajectory
    plt.figure(figsize=(5,5))
    plt.plot(x, y)
    for i in range(0, n_place_cells):
        curr_x_pos = x_spiking_positions[i]
        curr_y_pos = y_spiking_positions[i]
        for j in range(0, len(curr_x_pos)):
            plt.plot(curr_x_pos[j], curr_y_pos[j], 'ro')
    plt.show()