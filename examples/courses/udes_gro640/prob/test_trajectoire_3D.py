#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import DrillingRobot
from augn1301      import goal2r, r2q # Load your functions

# Define end-effector motion
r_0 = np.array([  0.5,   0.0,   1.0]) # start
r_f = np.array([ -0.25, -0.4,   0.6]) # end-point
t_f = 3.0                             # duration

# Compute the trajectory of the effector
r, dr, ddr = goal2r( r_0 , r_f , t_f )

n = r.shape[1] # Number of time steps

# Compute the trajectory of the joints
model      = DrillingRobot() # Robot Model
q, dq, ddq = r2q( r, dr, ddr , model )

t   = np.linspace(0, t_f, n)   # t
q1  = q[0,:]
q2  = q[1,:]
q3  = q[2,:]
dq1 = dq[0,:]
dq2 = dq[1,:]
dq3 = dq[2,:]

###################################################
# Direct joint trajectory test
#q1 = np.linspace(0,1, n)     # q1(t)
#q2 = np.linspace(1.4,2, n)   # q2(t)
#q3 = np.linspace(-1.3,2, n)  # q3(t)
###################################################


###################################################
# Visualisation of the trajectory

sys = DrillingRobot()
sys.compute_trajectory( t_f , n ) # little hack to use the plotting tool without simulations

sys.traj.t       = t
sys.traj.x[:,0]  = q1
sys.traj.x[:,1]  = q2
sys.traj.x[:,2]  = q3
sys.traj.x[:,3]  = dq1
sys.traj.x[:,4]  = dq2
sys.traj.x[:,5]  = dq3

# Visualise trajectory with animation
sys.animate_simulation( is_3d = True )

# Visualise joint trajectory 
sys.plot_trajectory('x')

# Visualise x-y-z trajectory of the end-effector
sys.plot_end_effector_trajectory()