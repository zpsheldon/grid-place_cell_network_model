# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:50:56 2018

Quadrant reversing trick for plotting inhibition and excitation kernels
Assumes square and even matrix for now
@author: ronwd
"""
import numpy as np

def quadflip (the_array):
    
    num_row=len(the_array);
    num_col=len(the_array[1]);
    q1_ul=np.array(the_array[:int(num_row/2),:int(num_col/2)]); #upper left
    q2_ur=np.array(the_array[:int(num_row/2),int(num_col/2):]); #upper right
    q3_lr=np.array(the_array[int(num_row/2):,int(num_col/2):]); #lower right
    q4_ll=np.array(the_array[int(num_row/2):,:int(num_col/2)]);#lower left
    
    q1_ul=np.reshape(q1_ul,(len(q1_ul),len(q1_ul[1])));
    q2_ur=np.reshape(q2_ur,(len(q2_ur),len(q2_ur[1])));
    q3_lr=np.reshape(q3_lr,(len(q3_lr),len(q3_lr[1])));
    q4_ll=np.reshape(q4_ll,(len(q4_ll),len(q4_ll[1])));
    
    #return q1_ul, q2_ur, q3_lr, q4_ll if just want to get quadrants
    
    q1_ul=np.flipud(np.fliplr(q1_ul))
    q2_ur=np.flipud(np.fliplr(q2_ur))
    q3_lr=np.flipud(np.fliplr(q3_lr))
    q4_ll=np.flipud(np.fliplr(q4_ll))
    
    q1q2=np.concatenate([q1_ul,q2_ur],1)
    q4q3=np.concatenate([q4_ll,q3_lr],1)
    
    new_array=np.concatenate([q1q2, q4q3],0)
    
    return new_array
    
    

