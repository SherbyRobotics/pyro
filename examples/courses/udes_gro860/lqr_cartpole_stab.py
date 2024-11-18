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

sys.u_ub[0] = +5
sys.u_lb[0] = -5

sys.xbar = np.array([0,np.pi,0,0]) # target state

################
# Cost function
################

cf  = costfunction.QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1.0
cf.Q[1,1] = 1.0
cf.R[0,0] = 1.0

###########################
# LQR Controller
##########################

cf.R[0,0] = 5.0

ctl = lqr.linearize_and_synthesize_lqr_controller( sys , cf )


###########################
# Simulation
##########################

cl_sys = ctl + sys

cl_sys.x0[0] = 0.5
cl_sys.x0[1] = 0.0
    
cl_sys.compute_trajectory( tf = 10.0 )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=1.0 )

