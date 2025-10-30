#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:04:51 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.pendulum import DoublePendulum
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation


sys = DoublePendulum()

# Max/Min torque
sys.u_ub[0] = +20
sys.u_ub[1] = +20
sys.u_lb[0] = -20
sys.u_lb[1] = -20

planner = DirectCollocationTrajectoryOptimisation(sys, 0.2, 20)

planner.x_start = np.array([-np.pi, 0, 0, 0])
planner.x_goal = np.array([0, 0, 0, 0])

planner.maxiter = 500
planner.set_linear_initial_guest(True)
planner.init_dynamic_plot()
planner.compute_optimal_trajectory()
# planner.show_solution()
planner.animate_solution()
