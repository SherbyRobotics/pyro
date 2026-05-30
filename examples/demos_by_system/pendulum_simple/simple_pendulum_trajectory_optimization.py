#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:04:51 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.pendulum import SinglePendulum
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation


sys = SinglePendulum()

# Max/Min torque
sys.u_ub[0] = +4
sys.u_lb[0] = -6

planner = DirectCollocationTrajectoryOptimisation(sys, grid=40, dt=0.1)

planner.x_start = np.array([0, 0])
planner.x_goal = np.array([-3.14, 0])

planner.init_dynamic_plot()
planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()
