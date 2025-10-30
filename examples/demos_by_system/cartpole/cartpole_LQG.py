#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic import cartpole
from pyro.control import controller
from pyro.dynamic import statespace
from scipy import linalg


###############################################################################
class LQG_Controller(controller.DynamicController):

    ############################
    def __init__(self, A, B, C, D, Qn, Rn):
        """ """

        # Dimensions
        self.k = 1
        self.m = 1
        self.p = 4
        self.l = 4

        super().__init__(self.k, self.l, self.m, self.p)

        # Label
        self.name = "Custom Cart-pole Controller"

        # Linear gain matrix

        self.K = np.array([-0.4472136, 26.64195636, -1.28268066, 5.80548107])

        # Observer gain matrix

        # Solve Riccati equation for observer
        P = linalg.solve_continuous_are(a=A.T, b=C.T, q=Qn, r=Rn)
        L = np.linalg.solve(Rn.T, (C @ P.T)).T

        self.L = L

        self.A = A  # np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = B  # np.array([[0], [0], [0], [1]])
        self.C = C  # np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.D = D  # np.array([[0], [0], [0], [0]])

    #############################
    def c(self, z, y, r, t=0):
        """Control law of the controller"""

        x_hat = z

        u = -self.K @ x_hat

        u = np.array([u])

        return u

    ############################
    def b(self, z, y, r, t):
        """Update law of internal controller states"""

        y = y - np.array([0.0, np.pi, 0.0, 0.0])

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        L = self.L
        K = self.K

        x_hat = z

        u = -K @ x_hat

        u = np.array([u])

        dx_hat = A @ x_hat + B @ u + L @ (y - C @ x_hat)

        return dx_hat


###############################################################################
sys = cartpole.CartPole()

sys.xbar[1] = np.pi  # Up-right position

ss = statespace.linearize(sys, 0.01)


Qn = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Rn = 0.001 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

ctl = LQG_Controller(ss.A, ss.B, ss.C, ss.D, Qn, Rn)

from pyro.dynamic import stochastic


Qn = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.01]])
Rn = 0.001 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

sys2 = stochastic.StochasticSystemWrapper(sys, Q=Qn, R=Rn, random_seed=42)


cl_sys = ctl + sys2

cl_sys.x0[0] = -3.0
cl_sys.x0[1] = 2.9

cl_sys.compute_trajectory(tf=4.0)
cl_sys.plot_trajectory("xu")
cl_sys.plot_trajectory("y")
cl_sys.plot_internal_controller_states()
cl_sys.animate_simulation(time_factor_video=0.5)
