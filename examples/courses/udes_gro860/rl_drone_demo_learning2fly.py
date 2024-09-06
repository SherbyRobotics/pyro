#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

#################################################################
### Dynamic
#################################################################

from pyro.dynamic.drone import Drone2D

class NormallizedDrone2D(Drone2D):

    def __init__(self):
        super().__init__()

        # Parameters
        self.mass = 1.0    # kg
        self.inertia = 0.1 # kgm2
        
        # Normalize inputs
        self.u_min = np.array([-1,-1])
        self.u_max = np.array([+1,+1])
        self.input_units = ['%', '%']
        self.weight = self.gravity * self.mass
        self.trust2weight = 1.2

        # Min/max states
        self.x_ub = np.array([10, 10, 2 * np.pi, 10, 10, 10])
        self.x_lb = -self.x_ub

    # Only change B actuation matrix in order to have normalized inputs values
    def f(self, x, u, t=0):

        # u = [0,0] : static gravity compensation trust
        # u = [1,1] : max trust 
        # u = [-1,-1] : min trust 
        u = self.weight * ( (self.trust2weight - 1) * u + np.array([0.5, 0.5]) )

        # from state vector (x) to angle and speeds (q,dq)
        [q, dq] = self.x2q(x)

        # compute joint acceleration
        ddq = self.ddq(q, dq, u, t)

        # from angle and speeds diff (dq,ddq) to state vector diff (dx)
        dx = self.q2x(dq, ddq)

        return dx

    def forward_kinematic_lines_plus(self, x, u, t):

        u = self.weight * ( (self.trust2weight - 1) * u + np.array([0.5, 0.5]) )

        return super().forward_kinematic_lines_plus(x, u, t)


sys = NormallizedDrone2D()

#################################################################
### Quick sim
#################################################################

sys.x0 = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.1])
sys.ubar = np.array([0.01, -0.01])
sys.compute_trajectory(tf=2.0, n=10000, solver="euler")
# sys.plot_trajectory('xu')

# sys.animate_simulation()


#################################################################
### Cost function
#################################################################

from pyro.analysis import costfunction

class CustomCostFunction( costfunction.CostFunction ):
    """
    J = int( g(x,u,y,t) * dt ) + h( x(T) , T )
    """

    ############################
    def __init__(self):

        super().__init__()

        self.INF = 1E3
        self.EPS = 1E-1

        # Target state
        self.x_target = np.array([ 0, 0, 0, 0, 0, 0])

        # Quadratic cost weights
        self.Q = np.diag([1., 1., 6., 0.1, 0.1, 0.1])
        self.R = np.diag([0.001,0.001])

        # Optionnal zone of zero cost if ||dx|| < EPS
        self.ontarget_check = False


    #############################
    def g(self, x, u, t):
        """ Quadratic additive cost """

        # Delta values with respect to target state
        dx = x - self.x_target

        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot(  u.T , np.dot(  self.R ,  u ) ) )

        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0

        return dJ

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """

        return 0.0
    
sys.cost_function = CustomCostFunction()
    

#################################################################
### Gymnasium and distribution of initial states
#################################################################

env = sys.convert_to_gymnasium( dt = 0.05 )

# Standard deviation of initial states
x0_std = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 0.2])

# Gaussian distribution of initial states
env.reset_mode = "gaussian"
env.x0_std = x0_std


#################################################################
### Learning quick demo of exploration
#################################################################

# # Demo learning with visual output
# env.render_mode = "human"

# # RL algorithm
# nn = PPO("MlpPolicy", env, verbose=1 ) #, tensorboard_log="./tensorboard_log/")
# nn.learn(total_timesteps=100)


#################################################################
### Headless learning
#################################################################

# RL algorithm
from stable_baselines3 import PPO
nn = PPO("MlpPolicy", env, verbose=1)
# nn.learn(total_timesteps=100000)
nn.learn(total_timesteps=2e5)


#################################################################
### Controller
#################################################################

# # Create controller object from RL model
from pyro.control.reinforcementlearning import stable_baseline3_controller

ppo_ctl = stable_baseline3_controller(nn)
ppo_ctl.plot_control_law(sys=sys, n=100, i = 2, j = 5 )

#################################################################
### Simulation
#################################################################

# # Closed-loop simulation
cl_sys = ppo_ctl + sys
cl_sys.x0 = np.array([-5.0, 0.0, 1.0, 0.0, 0.0, 0.0])
cl_sys.compute_trajectory(tf=20.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
ani = cl_sys.animate_simulation()


# # # Closed-loop simulation
# cl_sys = ppo_ctl + sys
# cl_sys.x0 = np.array([2.0, 5.0, 0.0, 0.0, 0.0, 0.0])
# cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
# # cl_sys.plot_trajectory("xu")
# ani = cl_sys.animate_simulation()
