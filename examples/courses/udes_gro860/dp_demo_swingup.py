#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic import pendulum
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.planning import discretizer

sys = pendulum.SinglePendulum()

sys.x_ub = np.array([+10, +10])
sys.x_lb = np.array([-10, -10])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, [201, 201], [11])

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([-3.14, 0])  # target
qcf.INF = 500

qcf.Q[0, 0] = 5.0
qcf.R[0, 0] = 1.0
# qcf.R[0, 0] = 1000.0

qcf.S[0, 0] = 10.0
qcf.S[1, 1] = 10.0


# qcf = costfunction.TimeCostFunction(sys.xbar)

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, qcf)
# dp = dprog.DynamicProgramming2DRectBivariateSpline(grid_sys, qcf)

# dp.solve_bellman_equation( animate_cost2go = True )

# dp.compute_steps(200)
# dp.plot_policy()

# dp.solve_bellman_equation( tol = 1)
dp.solve_bellman_equation(tol=0.1, animate_cost2go=True)
# dp.solve_bellman_equation( tol = 1 , animate_policy = True )
# dp.plot_cost2go(150)

# dp.animate_cost2go( show = False , save = True )
# dp.animate_policy( show = False , save = True )

dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

ctl = dp.get_lookup_table_controller()


# ctl.plot_control_law( sys = sys , n = 100)


# asign controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([0.0, 0.0])
cl_sys.compute_trajectory(10, 10001, "euler")
cl_sys.plot_trajectory("xu")
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
