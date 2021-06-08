# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:41:26 2018

@author: Alexandre
"""

###############################################################################
from pyro.analysis import Trajectory
from pyro.control  import controller



###############################################################################
class OpenLoopController( controller.StaticController ) :
    """  Open-loop controller based on trajectory solution  """
    ############################
    def __init__(self, trajectory   ):
        """ """
        
        # Sys
        self.trajectory = trajectory
        
        # Dimensions
        self.k = 1   
        self.m = trajectory.m
        self.n = trajectory.n

        if trajectory.y is not None:
            self.p = trajectory.y.shape[1]
        else:
            self.p = trajectory.x.shape[1]

        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Open Loop Controller'

    #############################
    def c( self , y , r , t  ):
        """  U depends only on time """
        
        u = self.trajectory.t2u( t )
        
        return u

    @staticmethod
    def load_from_file(name):
        traj = Trajectory.load(name)
        return OpenLoopController(traj)

    @property
    def time_final(self):
        return self.trajectory.time_final