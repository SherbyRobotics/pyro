#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:08:03 2024

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

from pyro.dynamic import pendulum
from pyro.control import controller
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys = pendulum.InvertedPendulum()

# Physical parameters of the prototype
sys.gravity = 9.8
sys.m1 = 0.300
sys.l1 = 0.4
sys.lc1 = 0.4 
sys.I1 = 0.0

max_torque = 3.0

sys.l_domain = 2 * sys.l1  # graphical domain

# Min/max state and control inputs
sys.x_ub = np.array([+2 * np.pi, +12])
sys.x_lb = np.array([-2 * np.pi, -12])
sys.u_ub = np.array([+max_torque])
sys.u_lb = np.array([-max_torque])

# Cost Function
sys.cost_function.xbar = np.array([0, 0])  # target
sys.cost_function.R[0, 0] = 0.0
sys.cost_function.Q[0, 0] = 1.0
sys.cost_function.Q[1, 1] = 0.0

# # DP solution
# from pyro.planning import discretizer
# from pyro.planning import dynamicprogramming

# grid_sys = discretizer.GridDynamicSystem(sys, [201, 201], [21])

# dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, sys.cost_function)

# dp.solve_bellman_equation(tol=0.01)
# dp.clean_infeasible_set()
# dp.plot_policy()
# dp.plot_cost2go(jmax=10)

# Learning
env = sys.convert_to_gymnasium(dt=0.05, render_mode=None)

env.clipping_states = True  # test

model = PPO("MlpPolicy", env, verbose=1)

from pyro.control.reinforcementlearning import stable_baseline3_controller

ppo_ctl = stable_baseline3_controller(model)


ppo_ctl.plot_control_law(sys=sys, n=100)
plt.show()
plt.pause(0.001)

n_time_steps = 250000
batches = 5
env.render_mode = None
for batch in range(batches):
    model.learn(total_timesteps=int(n_time_steps / batches))
    ppo_ctl.plot_control_law(sys=sys, n=100)
    plt.show()
    plt.pause(0.001)

# Animating rl closed-loop
cl_sys = ppo_ctl + sys

cl_sys.x0 = np.array([-3.14, -0.0])
cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation()
