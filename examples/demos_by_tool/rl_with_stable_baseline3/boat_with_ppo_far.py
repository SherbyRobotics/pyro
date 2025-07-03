#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np
from pyro.dynamic.boat import Boat2D
from stable_baselines3 import PPO

#################################################################
### Non-linear model
#################################################################


class NormallizedBoat2D(Boat2D):
    # Only change B actuation matrix in order to have normalized inputs values
    def B(self, q, u):

        B = np.zeros((3, 2))

        B[0, 0] = 8000.0
        B[1, 1] = 3000.0
        B[2, 1] = -self.l_t * 3000.0

        return B


sys = NormallizedBoat2D()

# Min/max state and control inputs
sys.x_ub = np.array([10, 10, 4 * np.pi, 10, 10, 10])
sys.x_lb = -sys.x_ub

# Min/max inputs are normalized, scalling is in the B matrix
sys.u_ub = np.array([1, 1])
sys.u_lb = np.array([-1, -1])

# Cost function
sys.cost_function.Q = np.diag([1, 1, 6.0, 0.1, 0.1, 1.0])
sys.cost_function.R = np.diag([0.001, 0.001])

# Distribution of initial states
sys.x0 = np.array([0.0, 0.0, 0.0, 0, 0, 0])
x0_std = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 0.2])

x0_lb = np.array([-8, -5, -0.1 * np.pi, 0.0, 0, 0])
x0_ub = np.array([-6, +5, +0.1 * np.pi, 0.0, 0, 0])

#################################################################
### Learning
#################################################################

# # Demo learning with visual output
# # Create gymnasium environment
# env = sys.convert_to_gymnasium()
# env.render_mode = "human"
# # env.reset_mode = "gaussian"
# env.x0_std = x0_std
# env.reset_mode = "uniform"
# env.x0_lb = x0_lb
# env.x0_ub = x0_ub

# # RL algorithm
# model2 = PPO("MlpPolicy", env, verbose=1)  # , tensorboard_log="./tensorboard_log/")
# model2.learn(total_timesteps=100)

# Headless learning
# Create gymnasium environment
env = sys.convert_to_gymnasium()
# env.reset_mode = "gaussian"
# env.x0_std = x0_std
env.reset_mode = "uniform"
env.x0_lb = x0_lb
env.x0_ub = x0_ub

# RL algorithm
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5e6)

# Create controller object from RL model
from pyro.control.reinforcementlearning import stable_baseline3_controller

ppo_ctl = stable_baseline3_controller(model)
ppo_ctl.plot_control_law(sys=sys, n=100)

# Closed-loop simulation
cl_sys = ppo_ctl + sys
cl_sys.x0 = np.array([-5.0, 5.0, 1.0, 2.0, 0.0, 0.0])
cl_sys.compute_trajectory(tf=30.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
ani = cl_sys.animate_simulation()
