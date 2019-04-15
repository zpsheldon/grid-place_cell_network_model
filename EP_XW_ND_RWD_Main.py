#!/usr/bin/env python 
'above is for running script from terminal'
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 09 11:36:34 2018

EP_XW_ND_Model

@author: ronwd, Zach Sheldon
"""

import pyfftw
import time
import numpy as np
import os
from scipy import stats, io
import matplotlib.pyplot as plt
import EP_XW_ND_RWD_SetupNetwork as SN
import EP_XW_ND_RWD_RunNetwork as RN
from place_cell import PlaceCell
import place_cell_utilities as PCU
import pickle
np.random.seed(1)

UNDEF = -999;

"****************************************************************************"
"***********************SETUP THE NETWORK************************************"
"****************************************************************************"
################PARAMETERS###################

# Network Basics
h = 1; # Number of depths, 1
n = 160; # number of neurons per side in grid tile,  160 in Louis paper, 128 in Alex's new, 90 in Alex OG
dt = 1.0 # in miliseconds #note code has dividing  by tau throughout so this works out to actually being 1ms , this is a holdover from a mistake reading Louis' model
tau = 10.0 # neuron time constant in ms
num_minutes = 20

# Place Cell Parameters
n_place_cells = 16 # number of place cells
sigma = 5.0 # standard deviation for Gaussian place fields
norm_const = 0.00636371 # normalization constant for place cells with sigma = 5.0
x_dim = 130 # x dimension for environment - cm
y_dim = 130 # y dimension for environment - cm
dist_thresh = 20 # min euclidean distance between place cell centers - cm
rand_weights_max=0.00222

#Recurrent Inhibition Parameters
wmag=2.4; #2.4 for rate model
lmin=12.5; # periodic conditions # 7.5 setting to value for medium sized grid fields since only one layer
lmax=15.; # 15 in paper
lexp=-1; ##-1 in paper
wshift = 1;

#Coupling Parameters (no coupling in this network, just kept for ease of using other functions)
umag=0#2.6 for rate model
urad=4.0;#8.0 for rate model
u_dv=0; # 1 corresponds to dorsal to ventral
u_vd=0; # 1 corresponds to ventral to dorsal

#Hippocampal Parameters
amag =.6; # now trying . 6 from Alex's paper, was 1;#trying .7 instead of .5 due to particularly low activity in network when checked #1 in rate model
falloff = 4.0; #4.0 in rate model
falloff_low = 2.0; #1-8-19 quick dialing in check...failed #inside this scaled radius hippocampal input is amg
falloff_high= UNDEF; #outside this scaled radius hippocampal input is 0

#Padding parameters for fourier transforms
periodic = 1; #to run periodic or aperiodic boundary conditions
if periodic == 0:
    pad_min = np.ceil(2*lmax)+wshift
    npad = int(n+pad_min)+1; 
    npfft=npad * (npad/2 + 1);
    print('aperiodic boundaries')
else: 
    pad_min = 0;
    npad = n;
    npfft = n * (n/2 + 1)
    print('periodic boundaries')

#Initial value for rates for grid units
rinit = 1e-3;  

"*******************************************************************"
"***************Run Network Setup***********************************"
"*******************************************************************"

# Trajectory Parameters/Get Trajectory Data
[x,y,x_ind,y_ind, x_of_bins,y_of_bins,vx,vy,spatial_scale, boundaries, time_ind, traj_filename] = SN.get_trajectory('Trajectory_Data_2_full',dt, num_minutes);
[x2,y2,x_ind2,y_ind2, x_of_bins2,y_of_bins2,vx2,vy2,spatial_scale2, boundaries2, time_ind2, traj_filename2] = SN.get_trajectory('Trajectory_Data_6_full',dt, num_minutes);

# update x and y to be interpolated 
x = x_ind * spatial_scale
y = y_ind * spatial_scale
x2= x_ind2 * spatial_scale2
y2 = y_ind2 * spatial_scale2

#inhibition length scales
l = SN.set_inhib_length(h,lexp,lmin,lmax)
l2 = SN.set_inhib_length(h,lexp,lmin,lmax)

# inhibitory kernel overall and for each directional subpopulation
[w, w_r, w_l, w_u,w_d] = SN.setup_recurrent(h,npad,l,wmag,wshift)
[w2, w_r2, w_l2, w_u2,w_d2] = SN.setup_recurrent(h,npad,l2,wmag,wshift)
a = SN.setup_input(amag,n,falloff_high,falloff_low,falloff)
a2 = SN.setup_input(amag,n,falloff_high,falloff_low,falloff)
[r, r_r, r_l, r_d, r_u, r_masks] = SN.setup_population(h,n,npad,rinit)
[r2, r_r2, r_l2, r_d2, r_u2, r_masks2] = SN.setup_population(h,n,npad,rinit)

# setup place to grid connections
w_pg_a = SN.setup_pg(1, n, n_place_cells, rand_weights_max)
w_pg_b = SN.setup_pg(1, n, n_place_cells, rand_weights_max)

############################ PLACE CELL SETUP ##########################

# create place cells for both maps 
place_cells = PCU.create_place_cells(n_place_cells, sigma, x_dim, y_dim, dist_thresh,map_b=True)
place_cells_a = place_cells[0:16]
place_cells_b = place_cells[16:32]

# create arrays for keeping track of spiking and overall activity 
place_cell_spiking_a, place_activity_a = np.zeros((n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))
place_cell_spiking_b, place_activity_b = np.zeros((n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))

PCU.plot_centers(place_cells_a, dist_thresh)
PCU.plot_centers(place_cells_b, dist_thresh)

# save place cells for testing simulation later
pickle_out = open("place_cells_A.pickle", "wb") 
pickle.dump(place_cells_a, pickle_out)
pickle_out.close()

pickle_out = open("place_cells_B.pickle", "wb") 
pickle.dump(place_cells_b, pickle_out)
pickle_out.close()

"****************************************************************************"
"****************************NETWORK DYNAMICS********************************"
"****************************************************************************"

nflow0 = 1000 #flow populations activity with no velocity
nflow1 = 5000 #flow populations activity with constant velocity

# system noise
rnoise = 0.0 
vgain = .4

still = 0 # no motion if true; clears vx and vy from loaded or simulated trajectory

if still:
    vx = 0
    vy = 0
    
vflow = .8 
theta_flow = UNDEF

## spiking simulation

spiking = 1 #set whether spiking model or rate model 0 or 1 respectively
if spiking == 0:
    print('rate coded network')
else:
    print('spike coded network')
spike = 0


#########################################################################
######## All functions used for flowing activity in the network###########
##########################################################################

def update_neuron_activity(r, r_r, r_l, r_d, r_u,r_fft_plan, r_ifft_plan, vx,vy,r_field,spike,spiking,itter,singleneuronrec,time_ind,sna_eachlayer,row_record,col_record):
    
    if umag==0 or u_vd==0 and u_dv==0: #run uncoupled network convolutions
        rwu_l=RN.convolve_no(r_fft_plan, r_ifft_plan,r_l, w_l,npad, h)
        rwu_u=RN.convolve_no(r_fft_plan, r_ifft_plan,r_u, w_u,npad, h)
        rwu_d=RN.convolve_no(r_fft_plan, r_ifft_plan,r_d, w_d,npad, h)
        rwu_r=RN.convolve_no(r_fft_plan, r_ifft_plan,r_r, w_r,npad, h)

    # calculate fields
    [r_l,r_field_l]=RN.calculate_field(r,r_l, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[0,:,:], a, 1.0-vgain*vx, h, n, npad, itter, 0);
    [r_r, r_field_r]=RN.calculate_field(r,r_r, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[1,:,:], a, 1.0+vgain*vx, h, n, npad, itter, 0);
    [r_u,r_field_u]=RN.calculate_field(r,r_u, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[2,:,:], a, 1.0+vgain*vy, h, n, npad, itter, 0);
    [r_d,r_field_d]=RN.calculate_field(r,r_d, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[3,:,:], a, 1.0-vgain*vy, h, n, npad, itter, 0);
            
    r_field = r_field_l + r_field_r + r_field_u + r_field_d; #fix error r_field not being updated uniquely for each of the directions correctly
        
    if rnoise>0.: 
        for k in range(0,h,1):
            for i in range(0, n,1):
                for j in range (0,n,1):
                    r_field[k][i][j] = r_field[k][i][j] + rnoise * (2*stats.uniform.rvs()-1); 
    # update fields and weights
    r=RN.update_activity_spike(r,r_field, spike, h, n, dt, tau, itter);

    if singleneuronrec: #get rate and spiking data for single units on each layer, hate nested if loops, but for now just leave it
        sna_eachlayer[:,:,itter]= RN.get_singleneuron_activity(sna_eachlayer,r, spike, spiking, itter, row_record, col_record);
    else:
        sna_eachlayer = -999;
        
    return r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer

def update_neuron_activity_with_place(r, r_r, r_l, r_d, r_u,r_fft_plan, r_ifft_plan, vx,vy,r_field,spike,spiking,itter,singleneuronrec,time_ind,sna_eachlayer,row_record,col_record, curr_place_activity,w_pg):
    
    if umag==0 or u_vd==0 and u_dv==0: #run uncoupled network convolutions
        rwu_l=RN.convolve_no(r_fft_plan, r_ifft_plan,r_l, w_l,npad, h)
        rwu_u=RN.convolve_no(r_fft_plan, r_ifft_plan,r_u, w_u,npad, h)
        rwu_d=RN.convolve_no(r_fft_plan, r_ifft_plan,r_d, w_d,npad, h)
        rwu_r=RN.convolve_no(r_fft_plan, r_ifft_plan,r_r, w_r,npad, h)

    # get current iteration's place cell activity
    curr_place_activity = curr_place_activity.reshape((16,1,1))
    p = np.sum((curr_place_activity*w_pg),0)
    
    # calculate fields
    [r_l,r_field_l]=RN.calculate_field(r,r_l, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[0,:,:], a, 1.0-vgain*vx, h, n, npad, itter, p);
    [r_r, r_field_r]=RN.calculate_field(r,r_r, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[1,:,:], a, 1.0+vgain*vx, h, n, npad, itter, p);
    [r_u,r_field_u]=RN.calculate_field(r,r_u, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[2,:,:], a, 1.0+vgain*vy, h, n, npad, itter, p);
    [r_d,r_field_d]=RN.calculate_field(r,r_d, rwu_l, rwu_r, rwu_d, rwu_u,r_masks[3,:,:], a, 1.0-vgain*vy, h, n, npad, itter, p);
            
    r_field = r_field_l + r_field_r + r_field_u + r_field_d; #fix error r_field not being updated uniquely for each of the directions correctly
        
    if rnoise>0.: 
        for k in range(0,h,1):
            for i in range(0, n,1):
                for j in range (0,n,1):
                    r_field[k][i][j] = r_field[k][i][j] + rnoise * (2*stats.uniform.rvs()-1); 
    # update fields and weights
    r=RN.update_activity_spike(r,r_field, spike, h, n, dt, tau, itter);
    w_pg =  RN.update_weights(n_place_cells,1,n,curr_place_activity,r,w_pg,itter);       
    
    if singleneuronrec: #get rate and spiking data for single units on each layer, hate nested if loops, but for now just leave it
        sna_eachlayer[:,:,itter]= RN.get_singleneuron_activity(sna_eachlayer,r, spike, spiking, itter, row_record, col_record);
    else:
        sna_eachlayer = -999;
        
    return r, r_field, r_l, r_u, r_d, r_r, sna_eachlayer, w_pg
                        

def flow_neuron_activity(v,theta,nflow,nphase,a,spike,spiking, r, r_r, r_l, r_d, r_u,singleneuronrec):
    print(nphase)
    vx = v *np.cos(theta)
    vy =  v * np.sin(theta)
    r_field = np.zeros((h,n,n))
            
    shapein = [npad, npad];
    shapeout = [npad, npad/2+1]
    r_fft_arrayin = pyfftw.empty_aligned(shapein, dtype = 'float64')
    r_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype = 'complex128')
    r_fft_plan = pyfftw.FFTW(r_fft_arrayin,r_fft_arrayout,axes =(0,1))
    r_ifft_plan = pyfftw.FFTW(r_fft_arrayout,r_fft_arrayin,axes =(0,1), direction = 'FFTW_BACKWARD')
            
    if singleneuronrec > 0:
        sna_eachlayer=np.zeros((h,3,time_ind))
        row_record = np.array([np.floor(np.size(r_field,1)/2) ,np.floor(np.size(r_field,1)/2)-10, np.floor(np.size(r_field,1)/2)+10  ])
        col_record =  np.array([np.floor(np.size(r_field,2)/2),np.floor(np.size(r_field,2)/2)-10 ,np.floor(np.size(r_field,2)/2)+10  ])
    else:
        row_record = col_record = sna_eachlayer = -999;
            
    for itter in range(1,nflow+1,1):
        # update neuron activity for grid cells with no place input
        [r,r_field, r_l, r_u, r_d, r_r, sna_eachlayer]  = update_neuron_activity(r, r_r, r_l, r_d, r_u, r_fft_plan, r_ifft_plan,vx,vy,r_field,spike,spiking,itter,singleneuronrec, nflow,sna_eachlayer,row_record,col_record)
            
    return r, r_field, r_r, r_l, r_d, r_u 
            
def flow_full_model(x, y, vx,vy,time_ind,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec, place_cell_spiking, place_activity,w_pg, place_cells):
    spike=np.zeros((h,n,n),dtype = 'int');
    shapein = [npad, npad];
    shapeout = [npad, npad/2+1]
    r_fft_arrayin = pyfftw.empty_aligned(shapein, dtype = 'float64')
    r_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype = 'complex128')
    r_fft_plan = pyfftw.FFTW(r_fft_arrayin,r_fft_arrayout,axes =(0,1))
    r_ifft_plan = pyfftw.FFTW(r_fft_arrayout,r_fft_arrayin,axes =(0,1), direction = 'FFTW_BACKWARD')
    r_field = np.zeros((h,n,n))

    if singleneuronrec > 0:
        sna_eachlayer=np.zeros((h,3,time_ind))
        row_record = np.array([np.floor(np.size(r_field,1)/2) ,np.floor(np.size(r_field,1)/2)-10, np.floor(np.size(r_field,1)/2)+10  ], dtype = 'int')
        col_record =  np.array([np.floor(np.size(r_field,2)/2),np.floor(np.size(r_field,2)/2)-10 ,np.floor(np.size(r_field,2)/2)+10  ], dtype='int')
    else:
        row_record = col_record = sna_eachlayer = -999;
         
    for itter in range(1,time_ind,1): 
        # update place cell activity matrices based on current velocity/position
        place_cell_spiking, place_activity = PCU.evaluate_spiking(place_cell_spiking, place_activity, place_cells, x, y, itter)
        curr_place_activity = place_activity[:, itter]
        curr_place_activity = curr_place_activity.reshape((1,16))
        if np.mod(itter,1000) == 0:
            print(itter)
        vx1 = vx[itter]
        vy1 = vy[itter]
        # update neuron activity for grid cells
        [r,r_field, r_l, r_u, r_d, r_r,sna_eachlayer, w_pg]  = update_neuron_activity_with_place(r, r_r, r_l, r_d, r_u,r_fft_plan, r_ifft_plan, vx1,vy1,r_field,spike,spiking,itter,singleneuronrec,time_ind,sna_eachlayer,row_record,col_record,curr_place_activity, w_pg)
        occ = 0; 
                
    return r, r_field, r_r, r_l, r_d, r_u,sna_eachlayer,occ, w_pg
    
"*************************Intialize Grid Activity and Place Connections**********************"
"********************************************************************************************"

# no velocity
status='Initializing Grid Activity in 4 Phases - Map A'
print(status)
singleneuronrec=False;
t0=time.time()
[r,r_field,r_r,r_l,r_d,r_u]=flow_neuron_activity(0,0,nflow0,1,a,spike,spiking, r, r_r, r_l, r_d, r_u,singleneuronrec)
# constant velocity
[r,r_field,r_r,r_l,r_d,r_u] = flow_neuron_activity(vflow, (np.pi/2 - np.pi/5) , nflow1,2,a,spike,spiking, r, r_r, r_l, r_d, r_u, singleneuronrec)

[r,r_field,r_r,r_l,r_d,r_u] = flow_neuron_activity(vflow, (2*np.pi/5), nflow1,3,a,spike,spiking, r, r_r, r_l, r_d, r_u,singleneuronrec)

[r,r_field,r_r,r_l,r_d,r_u] = flow_neuron_activity(vflow, (np.pi/4), nflow1,4,a,spike,spiking, r, r_r, r_l, r_d, r_u,singleneuronrec)
t_run1 = time.time()-t0 

singleneuronrec=True;
t0=time.time()
status='Running Model with Trajectory and Place Cells - Map A'
print(status)
[r, r_field, r_r, r_l, r_d, r_u, sna_eachlayer, occ, w_pg_a]=flow_full_model(x,y,vx, vy, time_ind, a, spike, spiking,r, r_r, r_l, r_d, r_u, singleneuronrec, place_cell_spiking_a, place_activity_a,w_pg_a, place_cells_a);
t_run2 = time.time()-t0 

# calculate single neuron spiking results
sns_eachlayer=np.zeros((sna_eachlayer.shape))
print('Calculating single neuron spiking results...')
for inds in range(0,np.size(sna_eachlayer,2),1):
    sns_eachlayer[:,:,inds] = sna_eachlayer[:,:,inds] > stats.uniform.rvs(0,1,size = h) 
    
sns_eachlayer=sns_eachlayer.astype('bool')

########################### PLOT RESULTS #####################################

# plot place fields
#print('Plotting place centers for map A...')
#plot_centers(place_cells_a, dist_thresh)

## plot grid cell results
#print('Plotting grid cell results...')
#for z in range(0,h,1):
#    plt.figure(4)
#    plt.imshow(r[z,:,:],cmap='hot')
#    plt.show()


# plot single neuron spiking results
#x_ind = np.reshape(x_ind,(x_ind.size, 1))
#y_ind = np.reshape(y_ind,(y_ind.size, 1))
#print('Plotting grid cell results over trajectory...')
#for z in range(0,3,1): 
#     plt.figure(z, figsize=(5,5))
#     plt.plot(x,y)
#     plt.plot(x_ind[sns_eachlayer[0,z,0:np.size(x_ind)]]*spatial_scale, y_ind[sns_eachlayer[0,z,0:np.size(x_ind)]]*spatial_scale,'r.')

# save variables
var_out = dict()
for i_vars in ('x', 'y', 'x_ind', 'y_ind', 'sna_eachlayer', 'sns_eachlayer', 'spatial_scale', 'r', 'w_pg_a'):
    var_out[i_vars] = locals()[i_vars]
cwd = os.getcwd()
io.savemat('grid_and_place_training_map_a_20min', var_out)

################################ Run model with second set of place cells #####################

# no velocity
status='Initializing Grid Activity in 4 Phases - Map B'
print(status)
singleneuronrec=False;
t0=time.time()
[r2,r_field2,r_r2,r_l2,r_d2,r_u2]=flow_neuron_activity(0,0,nflow0,1,a2,spike,spiking, r2, r_r2, r_l2, r_d2, r_u2,singleneuronrec)
# constant velocity
[r2,r_field2,r_r2,r_l2,r_d2,r_u2] = flow_neuron_activity(vflow, (np.pi/2 - np.pi/5) , nflow1,2,a2,spike,spiking, r2, r_r2, r_l2, r_d2, r_u2, singleneuronrec)

[r2,r_field2,r_r2,r_l2,r_d2,r_u2] = flow_neuron_activity(vflow, (2*np.pi/5), nflow1,3,a2,spike,spiking, r2, r_r2, r_l2, r_d2, r_u2,singleneuronrec)

[r2,r_field2,r_r2,r_l2,r_d2,r_u2] = flow_neuron_activity(vflow, (np.pi/4), nflow1,4,a2,spike,spiking, r2, r_r2, r_l2, r_d2, r_u2,singleneuronrec)
t_run1 = time.time()-t0 

singleneuronrec=True;
t0=time.time()
status='Running Model with Trajectory and Place Cells - Map B'
print(status)
[r2, r_field2, r_r2, r_l2, r_d2, r_u2, sna_eachlayer2, occ2, w_pg_b]=flow_full_model(x2,y2,vx2, vy2, time_ind2, a2, spike, spiking,r2, r_r2, r_l2, r_d2, r_u2, singleneuronrec, place_cell_spiking_b, place_activity_b,w_pg_b, place_cells_b);
t_run2 = time.time()-t0 

# calculate single neuron spiking results
sns_eachlayer2=np.zeros((sna_eachlayer2.shape))
print('Calculating single neuron spiking results...')
for inds in range(0,np.size(sna_eachlayer2,2),1):
    sns_eachlayer2[:,:,inds] = sna_eachlayer2[:,:,inds] > stats.uniform.rvs(0,1,size = h) 
    
sns_eachlayer2=sns_eachlayer2.astype('bool')

####################### PLOT RESULTS ###################################

# plot place cell results
#print('Plotting place centers for map B...')
#plot_centers(place_cells_b, dist_thresh)

## plot grid cell results
#print('Plotting grid cell results...')
#for z in range(0,h,1):
#    plt.figure(4)
#    plt.imshow(r2[z,:,:],cmap='hot')
#    plt.show()


# plot single neuron spiking results
#x_ind2 = np.reshape(x_ind2,(x_ind2.size, 1))
#y_ind2 = np.reshape(y_ind2,(y_ind2.size, 1))
#print('Plotting grid cell results over trajectory...')
#for z in range(0,3,1): 
#     plt.figure(z, figsize=(5,5))
#     plt.plot(x2,y2)
#     plt.plot(x_ind2[sns_eachlayer2[0,z,0:np.size(x_ind2)]]*spatial_scale2, y_ind2[sns_eachlayer2[0,z,0:np.size(x_ind2)]]*spatial_scale2,'r.')

# save variables
var_out_b = dict()
for i_vars in ('x2', 'y2', 'x_ind2', 'y_ind2', 'sna_eachlayer2', 'sns_eachlayer2', 'spatial_scale2', 'r2', 'w_pg_b'):
    var_out_b[i_vars] = locals()[i_vars]
cwd = os.getcwd()
io.savemat('grid_and_place_training_map_b_20min', var_out_b)

