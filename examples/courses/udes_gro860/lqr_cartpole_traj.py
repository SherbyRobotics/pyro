#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic   import cartpole
from pyro.analysis  import costfunction
from pyro.planning  import trajectoryoptimisation
from pyro.control   import lqr
###############################################################################

##############
# System
##############

sys  = cartpole.CartPole()

sys.u_ub[0] = +20
sys.u_lb[0] = -20

################
# Cost function
################

cf  = costfunction.QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1.0
cf.Q[1,1] = 1.0
cf.R[0,0] = 1.0

###########################
# Trajectory optimization
##########################

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( 
            sys , 
            dt = 0.1 , 
            grid = 40 ,
            cost_function = cf 
            )


planner.x_start = np.array([0,0,0,0])
planner.x_goal  = np.array([0,np.pi,0,0])

planner.init_dynamic_plot()
planner.maxiter = 500
planner.compute_optimal_trajectory()

traj = planner.traj


###########################
# LQR Controller
##########################

cf.R[0,0] = 5.0

ctl = lqr.TrajectoryLQRController( sys , traj , cf )


###########################
# Simulation
##########################

cl_sys = ctl + sys

cl_sys.x0[0] = 0.5
cl_sys.x0[1] = 0.5
    
cl_sys.compute_trajectory( tf = 10.0 )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=1.0 )

