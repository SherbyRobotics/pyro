#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import massspringdamper
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = massspringdamper.FloatingSingleMass()

sys.p = 2
sys.C = np.array([ [ 1 , 0 ],
                   [ 0 , 1 ]])
sys.D = np.array([ [ 0 ],
                   [ 0 ]])

sys.x_ub = np.array([+2, +2])
sys.x_lb = np.array([-2,  -2])
sys.u_ub = np.array([+1])
sys.u_lb = np.array([-1])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [3] )

# Cost Function
tcf = costfunction.TimeCostFunction( np.array([0.0,0.0]) ) 
tcf.INF = 10.0
tcf.EPS = 0.0001


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, tcf)

dp.solve_bellman_equation( tol = 0.0001 )


dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

# ctl = dp.get_lookup_table_controller()


# #asign controller
# cl_sys = ctl + sys
# cl_sys.x0   = np.array([1.2, 0.])
# cl_sys.compute_trajectory( 10, 10001, 'euler')
# cl_sys.plot_trajectory('xu')
# cl_sys.plot_phase_plane_trajectory()
# cl_sys.animate_simulation()
