#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.rocket import Rocket
from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace import linearize
from pyro.control.lqr import synthesize_lqr_controller


# Non-linear model
sys = Rocket()
sys.u_ub = np.array([2.0, 0.8]) * sys.mass * sys.gravity
sys.u_lb = np.array([0.0, -0.8]) * sys.mass * sys.gravity
sys.x0 = np.array([0.1,10,-0.1,0,0,0])

sys.inertia = 400

sys.xbar = np.array([0, 2.2, 0, 0, 0, 0])
sys.ubar = np.array([1, 0]) * sys.mass * sys.gravity  # Nominal trust = weight

# Linear model
ss = linearize(sys, 0.01)

# Cost function
cf = QuadraticCostFunction.from_sys(sys)
cf.Q[0, 0] = 1
cf.Q[1, 1] = 10000
cf.Q[2, 2] = 0.1
cf.Q[3, 3] = 0
cf.Q[4, 4] = 10000
cf.Q[5, 5] = 0
cf.R[0, 0] = 0.01
cf.R[1, 1] = 10.0

# LQR controller
ctl = synthesize_lqr_controller(ss, cf, sys.xbar, sys.ubar)

# Simulation Closed-Loop Non-linear with LQR controller
game = sys.convert_to_pygame(tf=10.0, dt=0.01, ctl=ctl, renderer="pygame")
