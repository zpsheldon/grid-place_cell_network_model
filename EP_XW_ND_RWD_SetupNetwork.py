# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 11:47:23 2018

EP_XW_ND_RWD_SetupNetwork

-Modified version of AK_LK_RWD model setup to account for running with less layers and to set up border cells to account for more walls
--May end up using this as general file or merging with AK_LK_RWD but for now keep it separate and then cross that bridge when we come to it


@author: ronwd
"""

import pyfftw
import numpy as np
from scipy import stats, io,signal,interpolate
import time
import quadflip as qp
import os

UNDEF=-999;

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_trajectory(traj_filename,dt, num_minutes):
    'NOTE THIS IS UDPATED ONE FROM 1/10/19 from AK_LK_RWD model that was now working'
    
    print('getting trajectory data from Hass lab')
    
    unpack=io.loadmat(traj_filename)
    times = num_minutes * 60; # convert to seconds
    traj_fs = unpack['traj_fs']
    
    #currently hard coding in the 5 since we only pulled 5 minutes of data from each session*********
    sesh_ts_og = np.arange(0,times,1/traj_fs); #time stamps (in seconds) for positions with original sampling rate
    sesh_ts_interp = np.arange(0,(times),(dt/1000)); #new time stamps (in seconds) for dt of model sampling rate
    
    spatial_scale = unpack['traj_spatial_scale']
    x = unpack['traj_x']*spatial_scale;#position in cm, below is in indecies, resample up to dt samples
    y = unpack['traj_y']*spatial_scale; #position in cm, below is in indecies, resample up to dt samples
  
    #dumb issue with grids, at least here not matching like with like aka loading full x and y even though simulation is for 5 minutes only
    # only take x's for time grabbed, this hopefully will fix issues with x_ind being full session when it should be 5 minutes
    x = x[0:np.size(sesh_ts_og)+1]
    y = y[0:np.size(sesh_ts_og)+1]
    
    #just to make sure we are all equal
    sesh_ts_og= sesh_ts_og[0:np.size(x)]
    
    #dumb fix to adjust for interpolation issues with some values of sesh_ts_interp being outside the range of sesh_ts_og
    sesh_ts_interp[sesh_ts_interp>np.max(sesh_ts_og)] = np.max(sesh_ts_og);

    if any(np.isnan(x)) or any(np.isnan(y)):
        
        nans1, t1= nan_helper(x)
        nans2, t2= nan_helper(y)
        
        x[nans1]= np.interp(t1(nans1), t1(~nans1), x[~nans1])
        y[nans2]= np.interp(t2(nans2), t2(~nans2), y[~nans2])
    
#    note hardcode issue
    x_ind = np.round(signal.resample(x/spatial_scale,int(times/(dt/1000)))); #resample up to dt samples
    x_ind = x_ind.astype('int');
    y_ind = np.round(signal.resample(y/spatial_scale,int(times/(dt/1000)))); # resample up to dt samples
    y_ind = y_ind.astype('int');
    
    #flipping around to make sure get all velocities we can (i.e. since pairing off need to count from back since first ind will be one without pair)
    x_temp =x[::-1]
    y_temp =y[::-1]
    
    vx_new = x_temp[0:-2:1]-x_temp[1:-1:1] 
    vx_new = np.append(vx_new, x_temp[1] - x_temp[0]) #get first subtraction back in
    vx_new = vx_new[::-1] #flip back around
    vx_new = np.append(0,vx_new,)*traj_fs; #append first zero

    vy_new = y_temp[0:-2:1]-y_temp[1:-1:1]
    vy_new = np.append(vy_new, y_temp[1] - y_temp[0]) #get first subtraction back in
    vy_new = vy_new[::-1] #flip back around
    vy_new = np.append(0,vy_new,)*traj_fs; #append first zero
    
    fvx = interpolate.interp1d(sesh_ts_og,vx_new[0,0:np.size(sesh_ts_og)]) 
    
    fvy = interpolate.interp1d(sesh_ts_og,vy_new[0,0:np.size(sesh_ts_og)])
    
    vx_new = fvx(sesh_ts_interp);
    vy_new = fvy(sesh_ts_interp);
    

    vx = vx_new/100.0; #to convert to M/S to align with other models
    vy = vy_new/100.0
    
    dim_box = np.array([1.0,1.0])*np.max([np.max(x), np.max(y)]) 
    boundaries = ([0,0],[dim_box[0], dim_box[1]])
    
    x_of_bins = np.arange(np.min(boundaries,0)[0],np.max(boundaries,0)[0]+spatial_scale,spatial_scale)
    y_of_bins = np.arange(np.min(boundaries,0)[1],np.max(boundaries,0)[1]+spatial_scale,spatial_scale)
    
    if np.any(x_ind>=np.size(x_of_bins)) or np.any(y_ind >= np.size(y_of_bins)):
        
        x_ind[x_ind>=np.size(x_of_bins)] = np.size(x_of_bins)-1;
        y_ind[x_ind>=np.size(y_of_bins)] = np.size(y_of_bins)-1;
  
    time_ind = int(times/(dt/1000)); #simply the number of indecies and therefore iterations for the model to run
    
    
    return x,y,x_ind,y_ind, x_of_bins,y_of_bins,vx,vy,spatial_scale, boundaries, time_ind, traj_filename
    
###############################################################################
"Initializes place to grid weights"

def setup_pg(h,n,n_place,rand_weights_max):
    w_pg = np.reshape(stats.uniform.rvs(0,0.00222,h*n*n*n_place),(n_place,n,n))
    return w_pg
    
###############################################################################
    
"Sets up inhibition radius across the layers"
def set_inhib_length(h,lexp,lmin,lmax):
    
    if h == 1:
        
        l = lmin;
        
    else:
        l=np.zeros(h); #initialize array for number of layers
    
        for k in range(0,h):
            zz =float(k) / float(h-1); #6/14/18 Update: have to change both to float since both start as int
        
            if lexp == 0. :
                l[k]=(lmin**1-zz) * (lmax**zz)
            
            else:
                l[k]=(lmin**lexp + (lmax**lexp - lmin**lexp)*zz)**(1/lexp) 
            
    return l
    
"General Equation for inhibition"

def w_value(x,y,l_value,wmag):
    
    r = np.sqrt(x*x+y*y);
    
    if r < 2 * l_value:
        w = -wmag/( 2*l_value*l_value) * (1 - np.cos(np.pi*r/l_value));
    else:
        w=0.;
        
    return w;
    
    
def setup_recurrent(h,npad,l,wmag,wshift):
    
    x=np.arange (-npad/2+1,npad/2+1)#o.g.  #for periodic (-npad/2,npad/2)
    y=np.arange (-npad/2+1,npad/2+1)#o.g.  #for periodic (-npad/2,npad/2)

    a_row=np.zeros((h,1,npad))
    a_column=np.zeros((h,npad,1))

    w=np.zeros((h,npad,npad))

    for k in range(0,h,1):
        for i in range (0,npad,1):
            for j in range (0,npad,1):
                w[k,i,j]=w_value(x[i],y[j],l,wmag);
                
    if wshift >0:
        
         w_ltemp=np.delete(w,0,2); w_ltemp=np.append(w_ltemp, a_column,2)
         w_rtemp=np.delete(w,npad-1,2); w_rtemp=np.append(a_column,w_rtemp,2)
         w_dtemp=np.delete(w,npad-1,1); w_dtemp=np.append(a_row,w_dtemp,1)
         w_utemp=np.delete(w,0,1); w_utemp=np.append(w_utemp,a_row,1)

         for k in range(0,h,1):
             
             w_rtemp[k,:,:]=qp.quadflip(w_rtemp[k,:,:])
             w_ltemp[k,:,:]=qp.quadflip(w_ltemp[k,:,:])
             w_utemp[k,:,:]=qp.quadflip(w_utemp[k,:,:])
             w_dtemp[k,:,:]=qp.quadflip(w_dtemp[k,:,:])
                
    else: 
        print ("Assuming no shift on inhibitory kernel!")
        w_rtemp = w_ltemp = w_utemp = w_dtemp = w;
        
        for k in range(0,h,1):
             
             w_rtemp[k,:,:]=qp.quadflip(w_rtemp[k,:,:])
             w_ltemp[k,:,:]=qp.quadflip(w_ltemp[k,:,:])
             w_utemp[k,:,:]=qp.quadflip(w_utemp[k,:,:])
             w_dtemp[k,:,:]=qp.quadflip(w_dtemp[k,:,:])
 
    shapein = [npad, npad];
    shapeout = [npad, npad/2+1]
    w_fft_arrayin = pyfftw.empty_aligned(shapein, dtype = 'float64')
    w_fft_arrayout = pyfftw.empty_aligned(shapeout, dtype = 'complex128')
    w_fft_plan = pyfftw.FFTW(w_fft_arrayin,w_fft_arrayout,axes =(0,1))

    w_r = np.zeros((h,npad,npad/2+1),dtype = 'complex128') 
    w_l = np.zeros((h,npad,npad/2+1),dtype = 'complex128') 
    w_u = np.zeros((h,npad,npad/2+1),dtype = 'complex128') 
    w_d = np.zeros((h,npad,npad/2+1),dtype = 'complex128') 
    
    for k in range(0,h,1):
        
        w_r[k,:,:] = w_fft_plan(w_rtemp[k,:,:])
        w_l[k,:,:] = w_fft_plan(w_ltemp[k,:,:])
        w_u[k,:,:] = w_fft_plan(w_utemp[k,:,:])
        w_d[k,:,:] = w_fft_plan(w_dtemp[k,:,:])
           
    return w, w_r, w_l, w_u,w_d
    
def setup_input(amag,n,falloff_high,falloff_low,falloff):

    a=np.zeros((n,n)); #initialize a to be size of neural sheet
    scaled=np.zeros(n)    
    
    for i in range(0,n,1): #iterate over the number of neurons
        scaled[i]=(i-n/2.+0.5) #/(n/2.); # sets up scaled axis with 0 at center

    for i in range(0,n,1): #iterate over full sheet which is nxn neurons
        for j in range (0,n,1):
            
            r = np.sqrt(scaled[i]*scaled[i]+scaled[j]*scaled[j])/(n/2.); #altered to align with paper, however they are equivalent
           
            if falloff_high != UNDEF and r >=falloff_high:
                a[i][j]=0 #If there is an upperlimit to spread and the spot on the sheet exceeds this, no hippocampal input is received

            elif falloff_low == UNDEF: #this is case in code currently
                if np.abs(r)<1:
                    a[i][j] = amag * np.exp(-falloff * r*r);
                else:
                    a[i][j]=0.0;
                    
            elif r <= falloff_low: 
            
                a[i][j] = amag;
            else:
                rshifted = r - falloff_low;
                a[i][j] = amag * np.exp(-falloff * rshifted*rshifted);
                
    return a 

def setup_population(h,n,npad,rinit):
    
        r=np.zeros((h,npad,npad))#just to multiply
        r_l=np.zeros((h,npad,npad))
        r_r=np.zeros((h,npad,npad))
        r_u=np.zeros((h,npad,npad))
        r_d=np.zeros((h,npad,npad))
        
        r_masks=np.zeros((4,npad,npad))# masks that will be used to get directional subpopulations out
        #order l, r, u, d
        r_masks[0,0:n:2,0:n:2]=1;
        r_masks[1,1:n:2,1:n:2]=1;
        r_masks[2,0:n:2,1:n:2]=1;
        r_masks[3,1:n:2,0:n:2]=1;
        
        for k in range (0,h,1):
            for i in range(0,n,2):
                    for j in range (0,n,2):
                       r[k][i][j] = rinit * stats.uniform.rvs();
            for i in range(0,n,2):
                    for j in range (1,n,2):
                        r[k][i][j] = rinit * stats.uniform.rvs();
            for i in range(1,n,2):
                    for j in range (0,n,2):
                        r[k][i][j] = rinit * stats.uniform.rvs();        
            for i in range(1,n,2):
                    for j in range (1,n,2): 
                        r[k][i][j] = rinit * stats.uniform.rvs();
        
        for k in range(0,h,1):
            r_l[k,:,:]=r[k,:,:]*r_masks[0,:,:] #elementwise multiplication
            r_r[k,:,:]=r[k,:,:]*r_masks[1,:,:]
            r_u[k,:,:]=r[k,:,:]*r_masks[2,:,:]
            r_d[k,:,:]=r[k,:,:]*r_masks[3,:,:]
            
        r=r[:,0:n,0:n] #remove zeros to get back to correct size for actual rate matrix
               
        return r, r_r, r_l, r_d, r_u, r_masks #return r_masks so don't have to recalculate each time
        
