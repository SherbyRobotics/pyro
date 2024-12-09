# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.analysis import simulation
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
###############################################################################

sys  = pendulum.DoublePendulum()

#Max/Min torque
sys.u_ub[0] = +20
sys.u_ub[1] = +20
sys.u_lb[0] = -20
sys.u_lb[1] = -20

#Planner
planner = DirectCollocationTrajectoryOptimisation( sys , 0.2 , 20 )

planner.x_start = np.array([3.14,0,0,0])
planner.x_goal  = np.array([0,0,0,0])

planner.maxiter = 500
planner.set_linear_initial_guest(True)
planner.compute_optimal_trajectory()

# Controller
ctl  = nonlinear.ComputedTorqueController( sys , planner.traj )

# goal
ctl.rbar = np.array([0,0])
ctl.w0   = 5
ctl.zeta = 1

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = planner.x_start 
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()