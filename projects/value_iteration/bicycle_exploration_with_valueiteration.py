# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

import numpy as np
###############################################################################
from pyro.dynamic import vehicle
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller
###############################################################################

sys  = vehicle.KinematicBicyleModel()

###############################################################################

# Set domain
sys.x_ub = np.array([+10,+10,+6.28])
sys.x_lb = np.array([-10,-10,-6.284])

sys.u_ub = np.array( [+1,+1.0] )
sys.u_lb = np.array( [-1,-1.0] )

grid_sys = discretizer.GridDynamicSystem( sys , (41,41,21) , (3,3) , 0.05 )

# Cost Function
cf = costfunction.QuadraticCostFunction(
    q=np.ones(sys.n),
    r=np.ones(sys.m),
    v=np.zeros(sys.p)
)

cf.xbar = np.array( [1, 2, 0] ) # target
cf.INF  = 1E4
cf.EPS  = 0
cf.R    = np.array([[0.01,0],[0,0]])

# VI algo

vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()
vi.load_data('bicycle_exp_vi')
vi.compute_steps(1, plot=True)
vi.save_data('bicycle_exp_vi')

vi.assign_interpol_controller()

vi.plot_cost2go()
vi.plot_policy(0)
vi.plot_policy(1)

# TEST: 3D policy showing
# vi.plot_3D_policy(0)
# vi.plot_3D_policy(1)
#
cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )
#
## Simulation and animation
x0   = [0.2,0.2,0]
tf   = 5

sim = cl_sys.compute_trajectory( x0 , tf , 10001 , 'euler')
cl_sys.get_plotter().plot(sim, 'xu')
cl_sys.get_animator().animate_simulation(sim, save=False, file_name='bicycle_exp')


# planner = randomtree.RRT( sys , x_start )
#
# speed    = 1
# steering = 0.5
#
# planner.u_options = [
#         np.array([ speed,-steering]),
#         np.array([ speed,+steering]),
#         np.array([ speed,0]),
#         np.array([ -speed,0])
#         ]
#
# planner.goal_radius = 1.0
# planner.dt          = 0.1
# planner.steps       = 5
#
# planner.find_path_to_goal( x_goal )
#
# planner.plot_tree()
# planner.plot_tree_3d()
# planner.plot_open_loop_solution()
#
# ###############################################################################
#
# sys.dynamic_domain = False
# sys.animate_simulation(planner.trajectory)