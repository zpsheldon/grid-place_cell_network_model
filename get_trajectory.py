#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:59:00 2018

@author: Ron DiTullio
"""

from scipy import io
import numpy as np
from scipy import interpolate
from scipy import signal

def get_trajectory(filename,dt, num_min):
    print('getting trajectory data from Hass lab')
    
    unpack=io.loadmat(filename)
    times = num_min * 60; # hardcoded because currently using only 5 minutes of each session
    traj_fs = unpack['traj_fs']
    
    #currently hard coding in the 5 since we only pulled 5 minutes of data from each session*********
    sesh_ts_og = np.arange(0,times,1/traj_fs); #time stamps (in seconds) for positions with original sampling rate
    sesh_ts_interp = np. arange(0,(times),(dt/1000)); #new time stamps (in seconds) for dt of model sampling rate
    
    #dumb fix to adjust for interpolation issues with some values of sesh_ts_interp being outside the range of sesh_ts_og
    sesh_ts_interp[sesh_ts_interp>np.max(sesh_ts_og)] = np.max(sesh_ts_og);
    
    spatial_scale = unpack['traj_spatial_scale']
    x = unpack['traj_x']*spatial_scale;#position in cm, below is in indecies, resample up to dt samples
    y = unpack['traj_y']*spatial_scale; #position in cm, below is in indecies, resample up to dt samples

    
#    note hardcode issue
    x_ind = np.round(signal.resample(unpack['traj_x'],int(times/(dt/1000)))); #resample up to dt samples
    x_ind = x_ind.astype('int');
    y_ind = np.round(signal.resample(unpack['traj_y'],int(times/(dt/1000)))); # resample up to dt samples
    y_ind = y_ind.astype('int');
    
    vx_new = np.append(0,(x[1:-1:1]-x[0:-2:1]))*traj_fs; 
    vx_new = np.append(vx_new,0);
    vy_new = np.append(0,(y[1:-1:1]-y[0:-2:1]))*traj_fs; 
    vy_new = np.append(vy_new,0);
    
    #get interpolation function for original sampling rate then use this to generate upsampled velocity data
    fvx = interpolate.interp1d(sesh_ts_og,vx_new) 
    
    fvy = interpolate.interp1d(sesh_ts_og,vy_new)
    
    vx_new = fvx(sesh_ts_interp);
    vy_new = fvy(sesh_ts_interp);
    

    vx = vx_new/100.0; #to convert to M/S to align with other models
    vy = vy_new/100.0
    
    #this is just from email with Jake, added a list of this later
    #armstrong 51 is current data, boundaries are at 1 m x 1 m
    dim_box = np.array([1.0,1.0])*100.0 #box is 1 meter by 1 meter, converted to cm
    boundaries = ([0,0],[dim_box[0], dim_box[1]])
    
    x_of_bins = np.arange(np.min(boundaries,0)[0],np.max(boundaries,0)[0]+spatial_scale,spatial_scale)
    y_of_bins = np.arange(np.min(boundaries,0)[1],np.max(boundaries,0)[1]+spatial_scale,spatial_scale)
    
    if np.any(x_ind>=np.size(x_of_bins)) or np.any(y_ind >= np.size(y_of_bins)):
        
        x_ind[x_ind>=np.size(x_of_bins)] = np.size(x_of_bins)-1;
        y_ind[x_ind>=np.size(y_of_bins)] = np.size(y_of_bins)-1;
     #note can shorten how much of even this slice of the trajectory you run by changing time_ind   
    time_ind = int(times/(dt/1000)); #simply the number of indecies and therefore iterations for the model to run
    
    
    return x,y,x_ind,y_ind, x_of_bins,y_of_bins,vx,vy,spatial_scale, boundaries, time_ind