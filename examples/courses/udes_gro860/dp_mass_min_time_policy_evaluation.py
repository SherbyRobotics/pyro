#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import massspringdamper
from pyro.analysis import costfunction
from pyro.control import controller
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer


# System
sys  = massspringdamper.FloatingSingleMass()

sys.p = 2
sys.C = np.array([ [ 1 , 0 ],
                   [ 0 , 1 ]])
sys.D = np.array([ [ 0 ],
                   [ 0 ]])

sys.x_ub = np.array([+2, +2])
sys.x_lb = np.array([-2,  -2])
sys.u_ub = np.array([+100]) # No torque limits for evaluation
sys.u_lb = np.array([-100]) # No torque limits for evaluation

# Cost Function
tcf = costfunction.TimeCostFunction( np.array([0.0,0.0]) ) 
tcf.INF = 20.0
tcf.EPS = 0.1

# Policy
class CustomController( controller.StaticController ) :
    
    ############################
    def __init__( self  ):
        """ """
        
        # Dimensions
        self.k   = 1 
        self.m   = 1
        self.p   = 2 
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Custom Controller'
        
    
    #############################
    def c( self , y , r , t = 0 ):

        position = y[0]
        velocity = y[1]
        
        u = - 0.5 * position - 0.5 * velocity

        # if (position<0):
        #     if ( velocity < np.sqrt(-2*position) ):
        #         u = 1
        #     else:
        #         u = -1
        # if (position>0):
        #     if ( velocity < -np.sqrt(2*position) ):
        #         u = +1
        #     else:
        #         u = -1

        u = np.clip(u,-1.0,1.0)

        return np.array([u])

ctl = CustomController()

ctl.plot_control_law( sys = sys , n = 100 )

# DP algo

grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [21] , 0.02, False )

dp = dynamicprogramming.PolicyEvaluatorWithLookUpTable(ctl, grid_sys, tcf)
# dp = dynamicprogramming.PolicyEvaluator(ctl, grid_sys, tcf)

dp.solve_bellman_equation( tol = 0.01 )
# dp.compute_steps( 500 )
dp.clean_infeasible_set()
dp.plot_cost2go_3D()


# Simulation
sys.cost_function = tcf
cl_sys = ctl + sys
cl_sys.x0   = np.array([0.5, 0.])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xuj')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
