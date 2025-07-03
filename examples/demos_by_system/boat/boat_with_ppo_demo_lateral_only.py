#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

from pyro.dynamic.boat import Boat2D

from stable_baselines3 import PPO

# from stable_baselines3 import TD3

import gymnasium as gym
from gymnasium import spaces

x_goal = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])


class BoatWithGate(Boat2D):

    # Change B actuation matrix in order to have normalized [-1,1] inputs values
    def B(self, q, u):

        B = np.zeros((3, 2))

        B[0, 0] = 8000.0
        B[1, 1] = 3000.0
        B[2, 1] = -self.l_t * 3000.0

        return B

    # Draw goal boat
    def forward_kinematic_lines_plus(self, x, u, t):

        lines_pts, lines_style, lines_color = super().forward_kinematic_lines_plus(
            x, u, t
        )

        # a, b, c = super().forward_kinematic_lines_plus(x_goal, u, t)

        # lines_pts = lines_pts + a
        # lines_style.append("-")
        # lines_color.append("r")

        return lines_pts, lines_style, lines_color


#################################################################
# Create a Gym Environment from a Pyro System
#################################################################
class Boat2Gym(gym.Env):
    """Create a Gym Environment from a Pyro System

    Taken from the Pyro system:
    - x0: nominal initial state
    - f: state evolution function xdot = f(x,u,t)
    - g: cost function g(x,u,t) (reward = -g)
    - h: observation function y = h(x,u,t)
    - x_ub, x_lb: state boundaries
    - u_ub, u_lb: control boundaries

    Additionnal parameters of the gym wrapper are:
    - dt: time step for the integration
    - tf: maximum duration of an episode
    - t0: initial time (only relevant if the system is time dependent)
    - render_mode: None or "human" for rendering the system
    - reset_mode: "uniform", "gaussian" or "determinist"
    - clipping_inputs: True if the system clips the control inputs
    - clipping_states: True if the system clips the states
    - x0_lb, x0_ub: boundaries for the initial state (only relevant if reset_mode is "uniform")
    - x0_std: standard deviation for the initial state (only relevant if reset_mode is "gaussian")
    - termination_is_reached: method to define the termination condition of the task (default is always False representing a continous time task)

    """

    metadata = {"render_modes": ["human"]}

    #################################################################
    def __init__(
        self, sys, dt=0.05, tf=10.0, t0=0.0, render_mode=None, reset_mode="uniform"
    ):

        # Boundaries
        self.t0 = t0
        self.tf = tf  # For truncation of episodes
        self.observation_space = spaces.Box(
            sys.x_lb, sys.x_ub
        )  # default is state feedback
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        # Dynamic evolution
        self.sys = sys
        self.dt = dt
        self.clipping_inputs = True  # The sys is assumed to clip out of bounds inputs
        self.clipping_states = False  # The sys is assumed to clip out of bounds states (some gym env do that but this is a very fake behavior not exibiited by real systems, to use with caution)

        # Rendering
        self.render_mode = render_mode

        # Reset parameters (stocasticity of the initial state)
        self.x0_lb = sys.x0 + 0.1 * sys.x_lb
        self.x0_ub = sys.x0 + 0.1 * sys.x_ub

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = t0

        # Init
        self.render_is_initiated = False

        if self.render_mode == "human":
            self.init_render()

    #################################################################
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.x = self.np_random.uniform(self.x0_lb, self.x0_ub)
        self.u = self.sys.ubar
        self.t = self.t0

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        return y, info

    #################################################################
    def step(self, u):

        # State and time at the beginning of the step
        x = self.x
        t = self.t
        dt = self.dt

        # Derivatives
        dx = self.sys.f(x, u, t)

        # Euler integration #TODO use a better integrator
        x_new = x + dx * dt
        t_new = t + dt

        # Horrible optionnal hack to avoid the system to go out of bounds
        if self.clipping_states:
            x_new = np.clip(x_new, self.sys.x_lb, self.sys.x_ub)

        # Termination of episodes
        terminated = self.termination_is_reached()

        # Reward = negative of cost function
        if terminated:
            r = -self.sys.cost_funtion.h(x, t)  # Terminal cost
        else:
            r = (
                -self.sys.cost_function.g(x, u, t) * dt
            )  # Instantaneous cost integrated over the time step

        # Truncation of episodes if out of space-time bounds
        truncated = (t_new > self.tf) or (not self.sys.isavalidstate(x_new))

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u  # The memory of the control input is only used for redering

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        # Rendering
        if self.render_mode == "human":
            self.render()

        return y, r, terminated, truncated, info

    #################################################################
    def init_render(self):

        self.render_is_initiated = True

        self.animator = self.sys.get_animator()
        self.animator.show_plus(self.x, self.u, self.t)
        plt.pause(0.001)

    #################################################################
    def render(self):

        if self.render_mode == "human":
            if not self.render_is_initiated:
                self.init_render()
            self.animator.show_plus_update(self.x, self.u, self.t)
            plt.pause(0.001)

    #################################################################
    def termination_is_reached(self):
        """This method should be overloaded in the child class to define the termination condition for a task that is not time defined in continous time."""

        # gate_is_reached = self.x[0] > 0.0

        return False


# Non-linear model
sys = BoatWithGate()

sys.x_ub = np.array([100, 100, 5 * np.pi, 100, 100, 10])
sys.x_lb = -np.array([100, 100, 5 * np.pi, 100, 100, 10])

sys.u_ub = np.array([1, 1])
sys.u_lb = np.array([-1, -1])

# Cost function
sys.cost_function.Q = np.diag([1, 1, 6.0, 0.1, 0.1, 1.0])
sys.cost_function.R = np.diag([0.001, 0.001])

# Distribution of initial states
x0_lb = np.array([-25, -5, -0.3 * np.pi, 0.0, 0, 0])
x0_ub = np.array([+5, +5, +0.3 * np.pi, 0.0, 0, 0])

# Demo
# env = Boat2Gym(sys, dt=0.1, tf=10.0, render_mode="human")
# env.x0_lb = x0_lb
# env.x0_ub = x0_ub
# model2 = PPO("MlpPolicy", env, verbose=1)
# model2.learn(total_timesteps=200)


# # ENV Fast learning
env = Boat2Gym(sys, dt=0.05, tf=10.0, render_mode=None)

# # Initial rest state boundaries
env.x0_lb = x0_lb
env.x0_ub = x0_ub

model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=500000)
model.learn(total_timesteps=5e6)


###############################################################################
import numpy as np

###############################################################################
from pyro.control.controller import StaticController

###############################################################################


class ppo_controller(StaticController):
    """
    Wrap a stable baseline 3 model to use it as a pyro controller
    """

    def __init__(self, model):

        self.model = model

        # Dimensions
        self.k = model.observation_space.shape[0]
        self.m = model.action_space.shape[0]
        self.p = model.observation_space.shape[0]

        StaticController.__init__(self, self.k, self.m, self.p)

        self.name = "Stable Baseline 3 Controller"

    #############################
    def c(self, y, r, t=0):
        """
        Feedback static computation u = c( y, r, t)

        INPUTS
        y  : sensor signal vector          p x 1
        r  : reference signal vector       k x 1
        t  : time                          1 x 1

        OUTPUTS
        u  : control inputs vector         m x 1

        """

        u, _x = self.model.predict(y, deterministic=True)

        return u


ppo_ctl = ppo_controller(model)
ppo_ctl.plot_control_law(sys=sys, n=100)

# # Animating rl closed-loop
cl_sys = ppo_ctl + sys
cl_sys.x0 = np.array([-15.0, -3.0, 0.0, 0.0, 0.0, 0.0])
cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
cl_sys.plot_trajectory("xu")
cl_sys.animate_simulation()
