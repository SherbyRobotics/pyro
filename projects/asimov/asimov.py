# -*- coding: utf-8 -*-
"""
Created on Mon July 22 2019

@author: Bruno-Pier Busque
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import manipulator
###############################################################################


###############################################################################
# 3D Asimov
###############################################################################

class Asimov( manipulator.ThreeLinkManipulator3D ):
    """
    Asimov Manipulator Class
    -------------------------------
    """

    ############################
    def __init__(self):
        """ """

        # initialize standard params
        super().__init__()

        # Name
        self.name = 'Asimov Manipulator'

        # Kinematic params
        self.l1 = 0.5  # [m]
        self.baseradius = 0.3  # [m]
        self.armradius = 0.05  # [m]
        self.l2 = 0.525  # [m]
        self.l3 = 0.375  # [m]
        self.lc1 = self.l1/2  # [m]
        self.lc2 = self.l2/2  # [m]
        self.lc3 = self.l3/2  # [m]
        self.lw = (self.l1 + self.l2 + self.l3)  # Total length

        # Inertia params
        self.m1 = 3.703  # [kg]
        self.mbras = 0.915  # [kg]
        self.mcoude = 0.832  # [kg]
        self.m2 = self.mbras + self.mcoude  # [kg]
        self.m3 = 0.576  # [kg]

        self.I1z = self.m1*(self.baseradius**2)/2

        self.I2x = (self.mbras*self.l2**2)/3 + self.mcoude*self.l2**2
        self.I2y = self.I2x
        self.I2z = self.m2*(self.armradius**2)/2

        self.I3x = (self.m3*self.l3**2)/3
        self.I3y = self.I3x
        self.I3z = self.m3*(self.armradius**2)/2

        # Labels, bounds and units
        self.x_ub = np.array([np.pi/2, -np.pi/4, np.pi/2])
        self.x_lb = np.array([-np.pi/2, -3*np.pi/4, -np.pi/2])


###############################################################################
# 2D Asimov
###############################################################################

class Asimov2D( manipulator.TwoLinkManipulator ):
    """
    Asimov 2D Manipulator Class
    -------------------------------
    A model of Asimov without the rotating base
    """

    ############################
    def __init__(self):
        """ """

        # initialize standard params
        super().__init__()

        # Name
        self.name = 'Asimov 2D Manipulator'

        self.l1 = 0.525  # [m]
        self.l2 = 0.375
        self.lc1 = self.l1/2
        self.lc2 = self.l2/2

        self.mbras = 0.915  # [kg]
        self.mcoude = 0.832  # [kg]
        self.m1 = self.mbras + self.mcoude  # [kg]
        self.m2 = 0.576  # [kg]
        self.I1 = (self.mbras*self.l1**2)/3 + self.mcoude*self.l1**2
        self.I2 = (self.m2*self.l2**2)/3


if __name__ == "__main__":
    """ MAIN TEST """

    sys = Asimov()
    dsys = manipulator.SpeedControlledManipulator.from_manipulator( sys )
    dsys.ubar = np.array([1, 1, 1])
    dsys.show3([0.1, 0.1, 0.1])
    dsys.x0 = np.array([0, 1, 1])  # Position initiale
    dsys.plot_trajectory('xu')
    dsys.animate_simulation( is_3d = True )

