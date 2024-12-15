# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np

##############################################################################
from pyro.dynamic import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming

##############################################################################

sys = pendulum.SinglePendulum()

sys.x_ub = np.array([+6, +6])
sys.x_lb = np.array([-9, -6])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, [301, 301], [21])

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([-3.14, 0])  # target
qcf.INF = 300

qcf.R[0, 0] = 1.0

qcf.S[0, 0] = 10.0
qcf.S[1, 1] = 10.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, qcf)

dp.solve_bellman_equation(tol=1)
# dp.solve_bellman_equation( tol = 1 , animate_cost2go = True )
# dp.solve_bellman_equation( tol = 1 , animate_policy = True )

# dp.animate_cost2go( show = False , save = True )
# dp.animate_policy( show = False , save = True )

dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

ctl = dp.get_lookup_table_controller()

# Simulation

game = sys.convert_to_pygame( tf=10.0, dt=0.01, ctl=ctl, renderer="pygame" )