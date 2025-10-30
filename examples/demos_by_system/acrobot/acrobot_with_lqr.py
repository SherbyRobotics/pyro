#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:51:18 2020

@author: alex
"""

import numpy as np

from pyro.dynamic.pendulum      import Acrobot
from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace    import linearize
from pyro.control.lqr           import synthesize_lqr_controller


# Non-linear model
sys = Acrobot()

# Linear model
ss  = linearize( sys , 0.1 )

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )
cf.Q[0,0] = 1.0
cf.Q[1,1] = 0.1
cf.Q[2,2] = 1.0
cf.Q[3,3] = 0.1
cf.R[0,0] = 0.001

# LQR controller
ctl = synthesize_lqr_controller( ss , cf )

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([-0.1,0.2,0,0])
cl_sys.compute_trajectory( tf = 5.0 )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=1.5 )
