# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import linear
###############################################################################


class SinglePendulum_with_position_output( pendulum.SinglePendulum ):
    
    def __init__(self):
        super().__init__()
        
        self.p    = 1             # output size
        #self.rbar = np.array([0]) # ref size


    def h(self, x, u, t):
        
        # New output function
        y = pendulum.SinglePendulum.h(self, x, u, t)
        
        y_position    = np.zeros(1)
        y_position[0] = y[0]
        
        return y_position

sys = SinglePendulum_with_position_output()
dof = 1

kp = 2 # 2,4
kd = 1 # 1
ki = 1

ctl = linear.PIDController(kp, ki, kd)

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0[0] = 1.0
cl_sys.compute_trajectory(tf=10, n=20001, solver='euler')
cl_sys.plot_phase_plane_trajectory()
cl_sys.plot_trajectory('xu')
cl_sys.plot_internal_controller_states()
cl_sys.animate_simulation()