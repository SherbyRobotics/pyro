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

# Physical parameters
sys.gravity = 10.0
sys.m1 = 1.0
sys.l1 = 1.0
sys.lc1 = 0.5 * sys.l1
sys.I1 = (1.0 / 12.0) * sys.m1 * sys.l1**2

sys.l_domain = 2 * sys.l1  # graphical domain

# Min/max state and control inputs
sys.x_ub = np.array([+2.0 * np.pi, +12])
sys.x_lb = np.array([-2.0 * np.pi, -12])
sys.u_ub = np.array([+8.0])
sys.u_lb = np.array([-8.0])

# Time constant
dt = 0.05

# Cost Function
# The reward function is defined as: r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
sys.cost_function.xbar = np.array([0, 0])  # target
sys.cost_function.R[0, 0] = 0.001 / dt
sys.cost_function.Q[0, 0] = 1.0 / dt
sys.cost_function.Q[1, 1] = 0.1 / dt

sys.x0 = np.array([-np.pi, 0.0])

# DP solution
from pyro.planning import discretizer
from pyro.planning import dynamicprogramming

grid_sys = discretizer.GridDynamicSystem(sys, [101, 101], [11])

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, sys.cost_function)

dp.solve_bellman_equation(tol=0.01)
dp.clean_infeasible_set()
dp.plot_policy()
dp.plot_cost2go_3D(jmax=5000)

# Learning
gym_env = sys.convert_to_gymnasium(dt=dt, render_mode=None)

gym_env.clipping_states = True # To reproduce the behavior of gym pendulum

gym_env.reset_mode = "uniform"
gym_env.x0_lb = np.array([-np.pi , -1.0])
gym_env.x0_ub = np.array([+np.pi , +1.0])

model = PPO("MlpPolicy", gym_env, verbose=1)

#model.load('pendulum_dp_vs_ppo_bangbang')

from pyro.control.reinforcementlearning import stable_baseline3_controller

ppo_ctl = stable_baseline3_controller(model)


ppo_ctl.plot_control_law(sys=sys, n=100)
plt.show()
plt.pause(0.001)

n_time_steps = 2.5E2
batches = 3
gym_env.render_mode = None
for batch in range(batches):
    model.learn(total_timesteps=int(n_time_steps / batches))
    ppo_ctl.plot_control_law(sys=sys, n=100)
    plt.show()
    plt.pause(0.001)

# Save the model
model.save('pendulum_dp_vs_ppo_bangbang')

# Animating rl closed-loop
cl_sys = ppo_ctl + sys

cl_sys.x0 = np.array([-3.2, -0.0])
cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation()
