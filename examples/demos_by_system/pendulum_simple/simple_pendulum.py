# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np

##############################################################################
from pyro.dynamic import pendulum

##############################################################################

# Dynamic system
sys = pendulum.SinglePendulum()

# Simulation
sys.x0 = np.array([0.7, 0])
sys.plot_trajectory("xu")
sys.plot_phase_plane_trajectory()
sys.animate_simulation()
