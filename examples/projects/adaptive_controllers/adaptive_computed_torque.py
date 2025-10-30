#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:15:49 2020

@author: alex
"""

import numpy as np
from pyro.control import controller


##############################################################################
class SinglePendulumAdaptativeController(controller.DynamicController):
    """
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """

    ############################
    def __init__(self, model, traj=None):
        """ """
        self.name = "Adaptive controller"

        # Params
        self.P = np.eye(2)
        self.K = 1
        self.lam = 1  # gain

        self.model = model

        k = model.dof
        m = model.m
        p = model.p

        l = 2  # number of states in the controller

        super().__init__(k, l, m, p)

        # Init internal states
        self.z0 = np.array([0.0, 0.0])

        self.internal_state_label = []
        self.internal_state_units = []

        for i in range(l):
            self.internal_state_label.append("a" + str(i))
            self.internal_state_units.append("")

    ############################
    def adaptative_variables(self, ddq_d, dq_d, q_d, dq, q):
        """

        Compute intermediate variables

        """
        q_e = q_d - q
        dq_e = dq_d - dq

        s = dq_e + self.lam * q_e
        dq_r = dq_d + self.lam * q_e
        ddq_r = ddq_d + self.lam * dq_e

        Y_r = np.zeros(2)

        Y_r[0] = ddq_r
        Y_r[1] = np.sin(q)

        return [s, dq_r, ddq_r, Y_r]

    ############################
    def b(self, a, x, r, t):
        """
        adaptation law

        """

        [q, dq] = self.model.x2q(x)

        [ddq_d, dq_d, q_d] = self.get_traj(t, r)

        [s, dq_r, ddq_r, Y_r] = self.adaptative_variables(ddq_d, dq_d, q_d, dq, q)

        da = self.P @ Y_r * s

        return da

    ############################
    def c(self, a, x, r, t=0):
        """

        control law

        """
        [q, dq] = self.model.x2q(x)

        [ddq_d, dq_d, q_d] = self.get_traj(t, r)

        [s, dq_r, ddq_r, Y_r] = self.adaptative_variables(ddq_d, dq_d, q_d, dq, q)

        u = Y_r @ a + self.K * s

        return u

    ############################
    def get_traj(self, t, r):
        """ """

        ddq_d = np.zeros(self.model.dof)
        dq_d = np.zeros(self.model.dof)
        q_d = r

        return [ddq_d, dq_d, q_d]


##############################################################################


class DoublePendulumAdaptativeController(controller.DynamicController):
    """ """

    ############################
    def __init__(self, model, traj=None):
        """ """

        self.name = "Adaptive controller"

        self.A = np.zeros(5)
        self.guess = np.zeros(5)
        self.T = np.eye(5)
        self.Kd = np.eye(2)
        self.lam = 1  # Sliding surface slope
        self.nab = 0.1  # Min convergence rate

        self.model = model

        k = model.dof
        m = model.m
        p = model.p
        l = 5

        super().__init__(k, l, m, p)

        # Init states
        self.z0 = np.zeros(5)

        self.internal_state_label = []
        self.internal_state_units = []

        for i in range(l):
            self.internal_state_label.append("a" + str(i))
            self.internal_state_units.append("")

    ##############################
    def trig(self, q):
        """
        Compute cos and sin usefull in other computation
        ------------------------------------------------

        """

        c1 = np.cos(q[0])
        s1 = np.sin(q[0])
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        c12 = np.cos(q[0] + q[1])
        s12 = np.sin(q[0] + q[1])

        return [c1, s1, c2, s2, c12, s12]

    ############################
    def adaptative_variables(self, ddq_d, dq_d, q_d, dq, q):
        """

        Given desired trajectory and actual state

        """
        q_e = q - q_d
        dq_e = dq - dq_d

        s = dq_e + self.lam * q_e
        dq_r = dq_d - self.lam * q_e
        ddq_r = ddq_d - self.lam * dq_e

        return [s, dq_r, ddq_r]

    ############################
    def adaptative_torque(self, Y, s, q, t):
        """

        Given actual state, compute torque necessarly to guarantee convergence

        """

        u_computed = np.dot(Y, self.A)

        u_discontinuous = np.dot(self.Kd, s)

        u_tot = u_computed - u_discontinuous

        return u_tot

    ############################
    def b(self, z, x, q_d, t):

        [q, dq] = self.model.x2q(x)

        ddq_d = np.zeros(self.model.dof)
        dq_d = np.zeros(self.model.dof)

        [s, dq_r, ddq_r] = self.adaptative_variables(ddq_d, dq_d, q_d, dq, q)

        [c1, s1, c2, s2, c12, s12] = self.trig(q)

        Y = np.zeros((2, 5))
        dz = np.zeros(5)

        Y[0, 0] = ddq_r[0] * c2
        Y[0, 1] = ddq_r[1] * c2
        Y[0, 2] = s2 * dq[1] * dq_r[0]
        Y[0, 3] = s2 * (dq[0] + dq[1]) * dq_r[1]
        Y[0, 4] = s1 + s12
        Y[1, 0] = ddq_r[0] * c2
        Y[1, 1] = ddq_r[1]
        Y[1, 2] = s2 * dq[0] * dq_r[0]
        Y[1, 3] = 0
        Y[1, 4] = s12

        b = np.dot(Y.T, s)
        dz = -1 * np.dot(self.T, b)

        return dz

    ############################
    def c(self, z, x, q_d, t=0):
        """

        Given desired fixed goal state and actual state, compute torques

        """

        [q, dq] = self.model.x2q(x)

        ddq_d = np.zeros(self.model.dof)
        dq_d = np.zeros(self.model.dof)

        [s, dq_r, ddq_r] = self.adaptative_variables(ddq_d, dq_d, q_d, dq, q)
        [c1, s1, c2, s2, c12, s12] = self.trig(q)

        Y = np.zeros((2, 5))

        Y[0, 0] = ddq_r[0] * c2
        Y[0, 1] = ddq_r[1] * c2
        Y[0, 2] = s2 * dq[1] * dq_r[0]
        Y[0, 3] = s2 * (dq[0] + dq[1]) * dq_r[1]
        Y[0, 4] = s1 + s12
        Y[1, 0] = ddq_r[0] * c2
        Y[1, 1] = ddq_r[1]
        Y[1, 2] = s2 * dq[0] * dq_r[0]
        Y[1, 3] = 0
        Y[1, 4] = s12

        self.A = self.guess + self.get_z_integral(z)

        u = self.adaptative_torque(Y, s, q, t)

        return u

    ############################
    def get_z_integral(self, z):
        """get intergral error internal states"""

        return z[: self.l]


##############################################################################
