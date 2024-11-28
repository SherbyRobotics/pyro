#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.boat          import Boat2D

from stable_baselines3 import PPO
# from stable_baselines3 import TD3

# Non-linear model
sys = Boat2D()
sys.mass = 0.12
sys.inertia = 0.2
sys.l_t = 2.0
sys.damping_coef = np.array([0,0,0])
sys.Cm_max = 0
sys.Cx_max = 0
sys.Cy_max = 0

sys.x_ub = np.array([10, 10, 2*np.pi, 10, 10, 10])
sys.x_lb = -sys.x_ub

sys.u_ub = np.array([1, 1])
sys.u_lb = np.array([-1, -1])

sys.cost_function.Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1])
sys.cost_function.R = np.diag([0.001,0.001])

# sys.ubar = np.array([1, 1])
# sys.compute_trajectory()
# sys.animate_simulation()

# # Demo learning with output
# env = sys.convert_to_gymnasium()
# env.render_mode = "human"
# model2 = PPO("MlpPolicy", env, verbose=1)
# model2.learn(total_timesteps=1000)

# Demo real learning
env = sys.convert_to_gymnasium()
# env.reset_mode = "gaussian"
# env.reset_mode = "uniform"
env.render_mode = None
model = PPO("MlpPolicy", env, verbose=1)
# model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
# model.learn(total_timesteps=12.5E6)

print('Reset mode: ', env.reset_mode)


from pyro.control.reinforcementlearning import stable_baseline3_controller
ppo_ctl = stable_baseline3_controller(model)
ppo_ctl.plot_control_law(sys=sys, n=100)

# # Animating rl closed-loop
cl_sys = ppo_ctl + sys
cl_sys.x0 = np.array([-0.0, -2.0, 0.0, 0.0, 0.0, 0.0])
cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation()
