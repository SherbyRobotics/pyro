# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic import vehicle_steering
from pyro.planning import randomtree

###############################################################################

sys = vehicle_steering.KinematicCarModelwithObstacles()

###############################################################################

# Planning
planner = randomtree.RRT(sys)

planner.x_start = np.array([0, 0, 0])
planner.x_goal = np.array([30, 0, 0])

# Set domain
sys.x_ub = np.array([+35, +3.5, +3])
sys.x_lb = np.array([-5, -2, -3])

# Control actions
speed = 2
steering = 0.3

planner.u_options = [
    np.array([speed, +steering]),
    np.array([speed, -steering]),
    np.array([speed, 0]),
    np.array([-speed, 0]),
]

planner.goal_radius = 1.0
planner.dt = 0.1
planner.steps = 5
planner.max_nodes = 10000
planner.max_distance_compute = 50000

planner.compute_solution()

planner.plot_tree()
planner.plot_tree_3d()
planner.show_solution()
planner.animate_solution()
