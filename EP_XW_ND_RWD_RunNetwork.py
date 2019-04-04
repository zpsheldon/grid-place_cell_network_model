# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 11:47:25 2018

EP_XW_ND_RWD_SetupNetwork

-Modified version of AK_LK_RWD model running network functions to account for running with less layers and to set up border cells to account for more walls
--May end up using this as general file or merging with AK_LK_RWD but for now keep it separate and then cross that bridge when we come to it


@author: ronwd
"""

import numpy as np
from scipy import stats
import time

"Convolution function if no coupling"
def convolve_no(r_fft_plan, r_ifft_plan, r_dir, w_dir, npad, h):
    
        rwu_dir=np.zeros((h,npad,npad))    
               
        for k in range(0,h,1):
           
            r_dir_fourier = r_fft_plan(r_dir[k,:,:]);
            
            rwu_dir[k,:,:] = r_ifft_plan(r_dir_fourier*w_dir[k,:,:])
            
        return rwu_dir
    
   
"Convolution function if ventral to dorsal coupling"
def convolve_vd(r_fft_plan, r_ifft_plan,r_dir,w_dir,u,npad,h): 
    
    rwu_dir=np.zeros((h,npad,npad))
    r_dir_fourier=np.zeros((h,npad,npad/2+1),dtype = 'complex128')
   
    for k in range(0,h,1):
        r_dir_fourier[k,:,:] = r_fft_plan(r_dir[k,:,:])    

    for k in range(0,h-1,1):
        rwu_dir[k,:,:]= r_ifft_plan((w_dir[k,:,:]*r_dir_fourier[k,:,:]+u*r_dir_fourier[k+1,:,:]))
    
    rwu_dir[h-1,:,:]=r_ifft_plan((w_dir[h-1,:,:]*r_dir_fourier[h-1,:,:]))
    
    return rwu_dir  

"Calculate field with hippocampal input after all rates are updated from convolution"      
def calculate_field(r,r_dir, rwu_l, rwu_r, rwu_d, rwu_u,r_masks, a, vgain_fac, h, n, npad, itter, p):
        r_temp=np.zeros((h,npad,npad))
        r_field_dir = np.zeros((h,n,n))
        r_temp[:,0:n,0:n]=r;

        r_dir = r_temp * r_masks
        
        r_field_mask = r_masks[0:n,0:n]
       
        r_field_dir[:,:,:]= (rwu_l[:,0:n,0:n] + rwu_r[:,0:n,0:n] + rwu_u[:,0:n,0:n] + rwu_d[:,0:n,0:n] + a * vgain_fac + p)*r_field_mask;
                  
        return r_dir, r_field_dir
        
"Update activity if using a rate system"   
def update_rate(r,r_field,dt,tau,n,h):
        
        dt_tau = dt / tau

        activity_mask = (r_field>0.)*1;
        r = r + (-r + r_field*activity_mask)*dt_tau
        
        return r

"Update activity if using a spiking based system"                                  
def update_activity_spike(r,r_field, spike, h, n, dt, tau, itter):
    "Update succesful to matrix multiplication, cuts iteration time in half or more!!!! "
    
    k = 500.0; #Originally 500, but scaling according to new dt/tau (which is essentially dt in this case)
    beta_grid =  0.1 #Originally 0.1, but scaled according to new k and dt/tau (which is essentially dt in this case)
    dt_tau= dt/tau; #dt divied by tau or c in Alex's paper 
    dt_inseconds = dt/1000; #8/23/18 dt is in seconds for Alex's model for determine if spiking, dt_tau is a ratio so maybe works out with this...
    alpha = 0.5 #scale factor from Alex's paper
    spike=np.zeros((h,n,n),dtype = 'int'); #initialize new spike array each time
    threshold=np.reshape(stats.uniform.rvs(0,1,size = h*n*n),(h,n,n)); #grab new thresholds
    
    activity_mask = (r_field>0.)*1; #use this to replace if r_field >0 loop
    
    spike = ((k*dt_inseconds*(r_field*activity_mask - beta_grid)) > threshold)*1  #check if spikes on any neurons on all layers

    r = r + (-r + r_field*activity_mask)*dt_tau + alpha * spike
    
    return r
          
'Updates the weights of the place to grid connections'
                    
def update_weights(n_place,h,n,p_activity,r,w_pg,itter):
    
    #Parameters
    lambda_lr = 0.00001 #learning rate
    epsilon_pg = 0.4
    r_new_shape = np.repeat(r,n_place,0)    
    
    w_pg += lambda_lr *((p_activity * r_new_shape)*(epsilon_pg - w_pg) - r_new_shape*w_pg)
        
    return w_pg
  
def get_singleneuron_activity(sna_eachlayer,r_field, spike, spiking, itter,row_record,col_record):
    
    sna_eachlayer[:,0,int(itter)] = r_field[:,row_record[0],col_record[0]];
    sna_eachlayer[:,1,int(itter)] = r_field[:,row_record[1],col_record[1]];
    sna_eachlayer[:,2,int(itter)] = r_field[:,row_record[2],col_record[2]];

    return sna_eachlayer[:,:,itter]
