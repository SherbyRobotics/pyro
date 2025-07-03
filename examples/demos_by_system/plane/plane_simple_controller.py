#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:38:30 2023

@author: alex
"""

import numpy as np

from pyro.dynamic import plane

from pyro.control import controller


###############################################################################


class PLaneController(controller.StaticController):

    ############################
    def __init__(self):
        """ """

        # Dimensions
        self.k = 1
        self.m = 2
        self.p = 6

        super().__init__(self.k, self.m, self.p)

        # Label
        self.name = "Plane Controller"

        self.v_ref = 40
        self.y_ref = 0.0

    #############################
    def c(self, y, r, t=0):
        """
        Feedback static computation u = c(y,r,t)

        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1

        OUPUTS
        u  : control inputs vector    m x 1

        """

        v = y[3]
        theta = y[2]
        y = y[1]

        T = +10 * (self.v_ref - v)

        theta_ref = +0.1 * (self.y_ref - y)

        delta = -0.5 * (theta_ref - theta)

        u = np.array([T, delta])

        return u


sys = plane.Plane2D()
sys.l_w = 0.0

sys.x0 = np.array([0, 0, 0.2, 15, 0, 0])

ctl = PLaneController()

cl_sys = ctl + sys

cl_sys.compute_trajectory(2)

cl_sys.plot_trajectory()
cl_sys.animate_simulation(time_factor_video=0.2)
