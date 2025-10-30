#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.drone import Drone2D
from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace import linearize
from pyro.control.lqr import synthesize_lqr_controller


# Non-linear model
sys = Drone2D()

sys.xbar = np.array([0, 0, 0.0, 0, 0, 0])
sys.ubar = np.array([0.5, 0.5]) * sys.mass * sys.gravity

# Linear model
ss = linearize(sys, 0.01)

# Cost function
cf = QuadraticCostFunction.from_sys(sys)
cf.R[0, 0] = 0.001
cf.R[1, 1] = 0.001

# LQR controller
ctl = synthesize_lqr_controller(ss, cf, sys.xbar, sys.ubar)

# Simulation Open-Loop Non-linear
sys.x0 = np.array([-10, -1, 0, 0, 0, 0.1])
sys.compute_trajectory(5)
sys.plot_trajectory("xu")
sys.animate_simulation()

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([-10, -1, 0, 0, 0, 0])
cl_sys.compute_trajectory(5)
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation()
