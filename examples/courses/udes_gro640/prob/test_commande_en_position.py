#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import LaserRobot

from augn1301      import CustomPositionController  # Empty template


# Model cinématique du robot
sys = LaserRobot()

# Contrôleur en position de l'effecteur standard
ctl = CustomPositionController( sys )

# Cible de position pour l'effecteur
ctl.rbar = np.array([0,-1])

# Dynamique en boucle fermée
clsys = ctl + sys

# Configurations de départs
clsys.x0 =  np.array([0,0.5,0])   # crash 
# clsys.x0 =  np.array([0,0.7,0]) # fonctionne

# Simulation
clsys.compute_trajectory()
clsys.plot_trajectory('xu')
clsys.animate_simulation()