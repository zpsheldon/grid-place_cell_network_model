#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:13:13 2019

@author: Zach Sheldon
"""
import pyfftw
from scipy import stats, io
import EP_XW_ND_RWD_SetupNetwork as SN
import EP_XW_ND_RWD_RunNetwork as RN
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(1234)
UNDEF = -999

######################## NETWORK PARAMETERS ####################

# Network Basics
h = 1; # Number of depths, 1
n = 160; # number of neurons per side in grid tile,  160 in Louis paper, 128 in Alex's new, 90 in Alex OG
dt = 1.0 # in milliseconds #note code has dividing  by tau throughout so this works out to actually being 1ms , this is a holdover from a mistake reading Louis' model
tau = 10.0 # neuron time constant in ms
num_minutes = 20 # total number of minutes of traj data to use
num_blocks = 2 # number of blocks to use for experiment

# Place Cell Parameters
n_place_cells = 16 # number of place cells
sigma = 5.0 # standard deviation for Gaussian place fields
x_dim = 130 # x dimension for environment - cm
y_dim = 130 # y dimension for environment - cm
dist_thresh = 20 # min euclidean distance between place cell centers - cm
rand_weights_max=0.00222

#Recurrent Inhibition Parameters
wmag=2.4; #2.4 for rate model
lmin=12.5; # periodic conditions # 7.5 setting to value for medium sized grid fields since only one layer
lmax=15.; # 15 in paper
lexp=-1; #-1 in paper
wshift = 1;

#Coupling Parameters (no coupling in this network, just kept for ease of using other functions)
umag=0 #2.6 for rate model
urad=4.0; #8.0 for rate model
u_dv=0; # 1 corresponds to dorsal to ventral
u_vd=0; # 1 corresponds to ventral to dorsal

#Hippocampal Parameters
amag =.6; # now trying . 6 from Alex's paper, was 1;#trying .7 instead of .5 due to particularly low activity in network when checked #1 in rate model
falloff = 4.0; #4.0 in rate model
falloff_low = 2.0; #1-8-19 quick dialing in check...failed #inside this scaled radius hippocampal input is amg
falloff_high= UNDEF; #outside this scaled radius hippocampal input is 0

#Padding parameters for fourier transforms 
pad_min = 0;
npad = n;
npfft = n * (n/2 + 1)
print('periodic boundaries')

#Initial value for rates for grid units
rinit = 1e-3;  

# hippocampal excitation and grid matrix
a = SN.setup_input(amag,n,falloff_high,falloff_low,falloff)
[r, r_r, r_l, r_d, r_u, r_masks] = SN.setup_population(h,n,npad,rinit)

#inhibition length scales
l = SN.set_inhib_length(h,lexp,lmin,lmax)

# inhibitory kernel overall and for each directional subpopulation
[w, w_r, w_l, w_u,w_d] = SN.setup_recurrent(h,npad,l,wmag,wshift)

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
spiking=1
print('spike coded network')
spike = 0

######################## LOAD VARIABLES #######################
print('Loading place cells...')
vars_a = io.loadmat('grid_and_place_simulation_map_a_20min')
w_pg_A = vars_a['w_pg_a']
vars_b = io.loadmat('grid_and_place_simulation_map_b_20min')
w_pg_B = vars_b['w_pg_B']

# check that boundaries are similar
[x,y,x_ind,y_ind, x_of_bins,y_of_bins,vx,vy,spatial_scale, boundaries8, time_ind, traj_filename] = SN.get_trajectory('Trajectory_Data_8_full',dt, num_minutes) # traj for experiment
# update x and y to be interpolated
x = x_ind * spatial_scale
y = y_ind * spatial_scale
# length of each block
block_length = int(time_ind/num_blocks) 

# load place cells
pickle_in = open("place_cells_A.pickle","rb")
place_cells_A = pickle.load(pickle_in)

pickle_inB = open("place_cells_B.pickle","rb")
place_cells_B = pickle.load(pickle_inB)

# create arrays for keeping track of spiking and overall activity 
place_cell_spiking_A, place_activity_A = np.zeros((n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))
place_cell_spiking_B, place_activity_B = np.zeros((n_place_cells, len(x))), np.zeros((n_place_cells, len(x)))

# plot gaussian place cell centers
def plot_centers(place_cells, dist_thresh):
    # check if any are within distance threshold
    for i in range(0, n_place_cells):
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
                curr_place_val = curr_place_val / 0.00611658 # normalized for sigma = 2.5
                p_centers[k][j] = p_centers[k][j] + curr_place_val
    plt.figure(figsize=(5,5))
    plt.imshow(p_centers, cmap='hot')
    plt.gca().invert_yaxis()
    
plot_centers(place_cells_A, dist_thresh)
plot_centers(place_cells_B, dist_thresh)

########################### FUNCTIONS FOR RUNNING MODEL ############

# updates place cell spiking and place activity matrices given spiking at current position x and y at current time index
def evaluate_spiking(place_cell_spiking, place_activity, place_cells, x, y, time_idx):
    for i in range(0, len(place_cells)):
        curr_place_cell = place_cells[i]
        place_cell_spiking[i, time_idx], place_activity[i, time_idx] = curr_place_cell.evaluate_spiking(x[time_idx], y[time_idx])
    return place_cell_spiking, place_activity

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

def flow_full_model_with_blocks(x, y, vx,vy,time_ind,num_blocks,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec,w_pg_A, place_cells_A, place_cell_spiking_A, place_activity_A, w_pg_B, place_cells_B, place_cell_spiking_B, place_activity_B):
    # arrays for fft
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
    
    # keep track of grid activity for each block
    r_exp_blocks = np.zeros((num_blocks,160,160))
    
    # run model
    for itter in range(1,time_ind,1): 
        if time_ind >= time_ind/2:
            mapA = False
        else:
            mapA = True 
        # print iteration
        if np.mod(itter,1000) == 0:
            print("Iteration: %s" % (itter,))
        # get current velocity
        vx1 = vx[itter]
        vy1 = vy[itter]
        if mapA:
            # get current spiking/activity
            place_cell_spiking_A, place_activity_A = evaluate_spiking(place_cell_spiking_A, place_activity_A, place_cells_A, x, y, itter)
            curr_place_activity_A = place_activity_A[:, itter]
            # reshape for use in update neuron activity function
            curr_place_activity_A = curr_place_activity_A.reshape((1,16)) 
            # update neuron activity for grid cells
            [r,r_field, r_l, r_u, r_d, r_r,sna_eachlayer, w_pg_A] = update_neuron_activity_with_place(r, r_r, r_l, r_d, r_u,r_fft_plan, r_ifft_plan, vx1,vy1,r_field,spike,spiking,itter,singleneuronrec,time_ind,sna_eachlayer,row_record,col_record,curr_place_activity_A, w_pg_A)
        else:
            # get current spiking/activity
            place_cell_spiking_B, place_activity_B = evaluate_spiking(place_cell_spiking_B, place_activity_B, place_cells_B, x, y, itter)
            curr_place_activity_B = place_activity_B[:, itter]
            # reshape for use in update neuron activity function
            curr_place_activity_B = curr_place_activity_B.reshape((1,16)) 
            # update neuron activity for grid cells
            [r,r_field, r_l, r_u, r_d, r_r,sna_eachlayer, w_pg_B] = update_neuron_activity_with_place(r, r_r, r_l, r_d, r_u,r_fft_plan, r_ifft_plan, vx1,vy1,r_field,spike,spiking,itter,singleneuronrec,time_ind,sna_eachlayer,row_record,col_record,curr_place_activity_B, w_pg_B)

        # grab r matrix after map changes
        if itter == time_ind/2 + 100:
            r_exp_blocks[0, :, :] = r[0, :, :]
        elif itter == time_ind - 1:
            r_exp_blocks[1, :, :] = r[0, :, :]
        
    return r, r_exp_blocks, sna_eachlayer,w_pg_A, place_cell_spiking_A, place_activity_A, w_pg_B, place_cell_spiking_B, place_activity_B

############################ RUN EXPERIMENT #######################

# initialize grid pattern (no traj)
status='Initializing Grid Activity in 4 Phases'
print(status)
singleneuronrec=False;
# no velocity
[r,r_field,r_r,r_l,r_d,r_u]=flow_neuron_activity(0,0,nflow0,1,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec)
# constant velocity
[r,r_field,r_r,r_l,r_d,r_u]= flow_neuron_activity(vflow, (np.pi/2 - np.pi/5) , nflow1,2,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec)

[r,r_field,r_r,r_l,r_d,r_u]= flow_neuron_activity(vflow, (2*np.pi/5), nflow1,3,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec)

[r,r_field,r_r,r_l,r_d,r_u]= flow_neuron_activity(vflow, (np.pi/4), nflow1,4,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec)

# run with trajectory using place cells and light switches
def run_light_switch_experiment(x,y,vx,vy,time_ind,num_blocks,a,spike,spiking,r,r_r,r_l,r_d,r_u,singleneuronrec,w_pg_A, place_cells_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cells_B, place_cell_spiking_B, place_activity_B):
    singleneuronrec=True
    status='Running Model with Trajectory, Place Cells, and Light Switches'
    print(status)
    # run model
    [r, r_exp_blocks, sna_eachlayer, w_pg_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cell_spiking_B, place_activity_B] = flow_full_model_with_blocks(x, y, vx,vy,time_ind,num_blocks,a,spike,spiking,r, r_r, r_l, r_d, r_u,singleneuronrec, w_pg_A, place_cells_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cells_B, place_cell_spiking_B, place_activity_B)
    
    return r, r_exp_blocks, sna_eachlayer, w_pg_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cell_spiking_B, place_activity_B

[r, r_exp_blocks, sna_eachlayer,w_pg_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cell_spiking_B, place_activity_B] = run_light_switch_experiment(x,y,vx,vy,time_ind,num_blocks,a,spike,spiking,r,r_r,r_l,r_d,r_u,singleneuronrec,w_pg_A, place_cells_A, place_cell_spiking_A, place_activity_A,w_pg_B, place_cells_B, place_cell_spiking_B, place_activity_B)

# convert sna_eachlayer into sna_eachlayer for each block
sna_eachlayer_blocks = np.zeros((num_blocks, 3, block_length))
for curr_block_num in range(0, num_blocks):
    start_block = block_length * curr_block_num
    end_block = start_block + block_length
    sna_eachlayer_blocks[curr_block_num,:,:] = sna_eachlayer[0, :, start_block:end_block]

###################### PLOT RESULTS ###############################

# plot activity level from r matrix
plt.figure(figsize=(5,5))
plt.imshow(r_exp_blocks[0], cmap='hot')
plt.title('Activity Matrix for Block 1')
plt.gca().invert_yaxis()

plt.figure(figsize=(5,5))
plt.imshow(r_exp_blocks[1], cmap='hot')
plt.title('Activity Matrix for Block 2')
plt.gca().invert_yaxis()

# calculate sns
sns_eachlayer=np.zeros((sna_eachlayer.shape))
print('Calculating overall single neuron spiking results...')
for inds in range(0,np.size(sna_eachlayer,2),1):
    sns_eachlayer[:,:,inds] = sna_eachlayer[:,:,inds] > stats.uniform.rvs(0,1,size = h) 
    
sns_eachlayer=sns_eachlayer.astype('bool')

# plot sns overall
x_ind = np.reshape(x_ind,(x_ind.size, 1))
y_ind = np.reshape(y_ind,(y_ind.size, 1))
"Changed so that have to alter second index to plot other recordings, in order 0 is dead center, 1 is down and left, 2 is up and right"
print('Plotting overall spiking results over trajectory...')
plt.figure(figsize=(5,5))
plt.plot(x,y)
plt.plot(x_ind[sns_eachlayer[0,0,0:np.size(x_ind)]]*spatial_scale, y_ind[sns_eachlayer[0,0,0:np.size(x_ind)]]*spatial_scale,'r.')

## plot sns for block 1
sna_eachlayer_block1 = sna_eachlayer_blocks[0,:,:]
sna_eachlayer_block1 = sna_eachlayer_block1.reshape((1,3,block_length))
sns_eachlayer_block1=np.zeros((1,3,block_length))
print('Calculating block 1 single neuron spiking results...')
for inds in range(0,block_length,1):
    sns_eachlayer_block1[:,:,inds] = sna_eachlayer_block1[:,:,inds] > stats.uniform.rvs(0,1,size = h) 
    
sns_eachlayer_block1=sns_eachlayer_block1.astype('bool')

x_ind = np.reshape(x_ind,(x_ind.size, 1))
y_ind = np.reshape(y_ind,(y_ind.size, 1))
#"Changed so that have to alter second index to plot other recordings, in order 0 is dead center, 1 is down and left, 2 is up and right"
print('Plotting grid cell results over trajectory for block 1...')
x1 = x[:block_length]
y1 = y[:block_length]
x_ind1 = x_ind[:block_length]
y_ind1 = y_ind[:block_length]
plt.figure(figsize=(5,5))
plt.plot(x1,y1)
plt.plot(x_ind1[sns_eachlayer_block1[0,0,0:np.size(x_ind1)]]*spatial_scale, y_ind1[sns_eachlayer_block1[0,0,0:np.size(x_ind1)]]*spatial_scale,'r.')

## plot sns for block 2
sna_eachlayer_block2 = sna_eachlayer_blocks[1,:,:]
sna_eachlayer_block2 = sna_eachlayer_block2.reshape((1,3,block_length))
sns_eachlayer_block2=np.zeros((1,3,block_length))
print('Calculating single neuron spiking results for block 2...')
for inds in range(0,block_length,1):
    sns_eachlayer_block2[:,:,inds] = sna_eachlayer_block2[:,:,inds] > stats.uniform.rvs(0,1,size = h) 
    
sns_eachlayer_block2=sns_eachlayer_block2.astype('bool')

#"Changed so that have to alter second index to plot other recordings, in order 0 is dead center, 1 is down and left, 2 is up and right"
print('Plotting grid cell results over trajectory for block 2...')
x2 = x[block_length:block_length*2]
y2 = y[block_length:block_length*2]
x_ind2 = x_ind[block_length:block_length*2]
y_ind2 = y_ind[block_length:block_length*2]
plt.figure(figsize=(5,5))
plt.plot(x2,y2)
plt.plot(x_ind2[sns_eachlayer_block2[0,0,0:np.size(x_ind2)]]*spatial_scale, y_ind2[sns_eachlayer_block2[0,0,0:np.size(y_ind2)]]*spatial_scale,'r.')

# save variables/arrays
var_out = dict()
for i_vars in ('x', 'y', 'x_ind', 'y_ind', 'sna_eachlayer', 'sns_eachlayer', 'spatial_scale', 'r'):
    var_out[i_vars] = locals()[i_vars]
cwd = os.getcwd()
io.savemat('grid_and_place_20min_1switch', var_out)


