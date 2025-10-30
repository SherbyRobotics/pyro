# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.control import robotcontrollers
from pyro.dynamic import manipulator

###############################################################################

torque_controlled_robot = manipulator.OneLinkManipulator()

# Target
q_desired = np.array([0.5])

# Joint PID

dof = 1

joint_pd = robotcontrollers.JointPD(dof)
joint_pd.rbar = q_desired
joint_pd.kp = np.array([25])
joint_pd.kd = np.array([10])

# Closed-loops

robot_with_joint_pd = joint_pd + torque_controlled_robot

# Simulations
tf = 4
robot_with_joint_pd.x0 = np.array([0, 0])
robot_with_joint_pd.compute_trajectory(tf)
robot_with_joint_pd.plot_trajectory("xu")
robot_with_joint_pd.animate_simulation()
