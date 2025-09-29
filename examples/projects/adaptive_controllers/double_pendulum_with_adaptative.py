# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic import pendulum

# from adaptive_computed_torque import DoublePendulumAdaptativeController
###############################################################################

sys = pendulum.DoublePendulum()
sys.cost_function = None
ctl = DoublePendulumAdaptativeController(sys)

# Param adapt-control

ctl.z0[0] = 0
ctl.z0[1] = 0
ctl.z0[2] = 0
ctl.z0[3] = 0
ctl.z0[4] = 0

ctl.Kd[0, 0] = 5
ctl.Kd[1, 1] = 8

ctl.lam = 1.2

ctl.T[0, 0] = 12
ctl.T[1, 1] = 12
ctl.T[2, 2] = 12
ctl.T[3, 3] = 12
ctl.T[4, 4] = 12

# Set Point
ctl.rbar = np.array([0, 0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0[0] = 3.14
cl_sys.x0[1] = 2.0

cl_sys.state_label[4] = "H1"
cl_sys.state_label[5] = "H2"
cl_sys.state_label[6] = "C1"
cl_sys.state_label[6] = "C2"
cl_sys.state_label[8] = "g"

cl_sys.compute_trajectory(tf=10, n=20001, solver="euler")
cl_sys.plot_phase_plane_trajectory()
cl_sys.plot_trajectory("xu")
cl_sys.plot_internal_controller_states()
cl_sys.animate_simulation()
