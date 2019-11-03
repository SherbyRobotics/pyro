# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
import argparse
###############################################################################
from pyro.dynamic  import vehicle
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller
###############################################################################

sys  = vehicle.KinematicCarModelwithObstacles()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+35,+3.5,+3])
sys.x_lb = np.array([-5,-2,-3])

sys.u_ub = np.array( [+1,+1.0] )
sys.u_lb = np.array( [-1,-1.0] )

# Discrete world
grid_sys = discretizer.GridDynamicSystem( sys , (91, 91, 21) , (3,3) , 0.05 )
# Cost Function
cf = costfunction.QuadraticCostFunction(
    q=np.ones(sys.n),
    r=np.ones(sys.m),
    v=np.zeros(sys.p)
)

cf.xbar = np.array( [30, 0, 0] ) # target
cf.INF  = 1E4
cf.EPS  = 0.2
cf.R    = np.array([[0.01,0],[0,0]])

# VI algo

vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()
if load_data:
    vi.load_data('car_vi')
vi.compute_steps(100, plot=True)
if save_data:
    vi.save_data('car_vi')

vi.assign_interpol_controller()

vi.plot_cost2go()
vi.plot_policy(0)
vi.plot_policy(1)

cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )
#
## Simulation and animation
x0   = [0, 0 ,0]
tf   = 5

sim = cl_sys.compute_trajectory( x0 , tf , 10001 , 'euler')
cl_sys.get_plotter().plot(sim, 'xu')
cl_sys.get_animator().animate_simulation(sim, save=True, file_name='car')