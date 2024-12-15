#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic import pendulum
from pyro.control import controller
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.planning import discretizer

sys = pendulum.DoublePendulum()

sys.x_ub[0] = +1.0
sys.x_lb[0] = -5.0
sys.x_ub[2] = +5.0
sys.x_lb[2] = -5.0
sys.x_ub[1] = +1.0
sys.x_lb[1] = -5.0
sys.x_ub[3] = +5.0
sys.x_lb[3] = -5.0

sys.u_ub[0] = +10.0
sys.u_lb[0] = -10.0
sys.u_ub[1] = +10.0
sys.u_lb[1] = -10.0

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, [51, 31, 51, 31], [3, 3], dt=0.1)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([0, 0, 0, 0])  # target

qcf.Q[0, 0] = 1.0
qcf.Q[1, 1] = 1.0
qcf.Q[2, 2] = 0.1
qcf.Q[3, 3] = 0.1
qcf.R[0, 0] = 0.01
qcf.R[1, 1] = 0.01

qcf.INF = 3000
qcf.EPS = 1.0

tcf = costfunction.TimeCostFunction(np.array([0, 0, 0, 0]))
tcf.EPS = 0.5


# DP algo

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, qcf)

# dp.solve_bellman_equation(tol=1.0)
dp.compute_steps(100)

dp.plot_cost2go()
dp.plot_policy()


# asign controller
ctl = dp.get_lookup_table_controller()
cl_sys = controller.ClosedLoopSystem(sys, ctl)

##############################################################################

# Simulation and animation
cl_sys.x0 = np.array([-np.pi, 0, 0, 0])
cl_sys.compute_trajectory(30, 10001, "euler")
cl_sys.plot_trajectory("xu")
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
