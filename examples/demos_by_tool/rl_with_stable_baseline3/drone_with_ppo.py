#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.drone import Drone2D
from stable_baselines3 import PPO

#################################################################
### Non-linear model
#################################################################

class NormallizedDrone2D( Drone2D ): 

    # Only change B actuation matrix in order to have normalized inputs values
    def f(self, x , u , t = 0 ):

        u = 2.0 * u + self.gravity * self.mass * np.array([0.5, 0.5])

        # from state vector (x) to angle and speeds (q,dq)
        [ q , dq ] = self.x2q( x )       
        
        # compute joint acceleration 
        ddq = self.ddq( q , dq , u , t ) 
        
        # from angle and speeds diff (dq,ddq) to state vector diff (dx)
        dx = self.q2x( dq , ddq )        
        
        return dx
    
    def forward_kinematic_lines_plus(self, x, u, t):

        u = 0.2 * ( 2.0 * u + self.gravity * self.mass * np.array([0.5, 0.5]) )

        return super().forward_kinematic_lines_plus(x, u, t)
    
sys = NormallizedDrone2D()

# Min/max state and control inputs
sys.x_ub = np.array([10, 10, 2*np.pi, 10, 10, 10])
sys.x_lb = -sys.x_ub

# Min/max inputs are normalized, scalling is in the B matrix
sys.u_ub = np.array([1, 1])
sys.u_lb = np.array([-1, -1])

# Cost function
sys.cost_function.Q = np.diag([1., 1., 6., 0.1, 0.1, 0.1])
sys.cost_function.R = np.diag([0.001,0.001])

# Distribution of initial states
sys.x0 = np.array([0.0, 0.0, 0.0, 0, 0, 0])
x0_std = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 0.2])

#################################################################
### Learning
#################################################################

# Demo learning with visual output
# Create gymnasium environment
# env = sys.convert_to_gymnasium()
# env.render_mode = "human"
# env.reset_mode = "gaussian" 
# env.x0_std = x0_std
# # RL algorithm 
# model2 = PPO("MlpPolicy", env, verbose=1 ) #, tensorboard_log="./tensorboard_log/")
# model2.learn(total_timesteps=100)

# Headless learning
# Create gymnasium environment
env = sys.convert_to_gymnasium()
env.reset_mode = "gaussian" 
env.x0_std = x0_std
# RL algorithm 
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2E5)

# Create controller object from RL model
from pyro.control.reinforcementlearning import stable_baseline3_controller
ppo_ctl = stable_baseline3_controller(model)
ppo_ctl.plot_control_law(sys=sys, n=100)

# Closed-loop simulation
cl_sys = ppo_ctl + sys
cl_sys.x0 = np.array([-5.0, 0.0, 1.0, 0.0, 0.0, 0.0])
cl_sys.compute_trajectory(tf=20.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
ani = cl_sys.animate_simulation()
