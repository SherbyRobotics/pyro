#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic import cartpole
from pyro.analysis import costfunction
from pyro.control import lqr

###############################################################################


# sys
sys = cartpole.CartPole()

sys.xbar[1] = np.pi  # Up-right position


# Cost function
cf = costfunction.QuadraticCostFunction.from_sys(sys)

cf.Q[0, 0] = 100.0
cf.Q[1, 1] = 1.0
cf.R[0, 0] = 100

ctl = lqr.linearize_and_synthesize_lqr_controller(sys, cf)


# Simulation
cl_sys = ctl + sys

cl_sys.x0[0] = -3.0
cl_sys.x0[1] = 1.0

cl_sys.compute_trajectory(tf=0.3, solver="euler")
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation(time_factor_video=1.0)
