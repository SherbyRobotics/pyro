#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:04:53 2020

@author: alex
"""

import numpy as np

from pyro.dynamic import ContinuousDynamicSystem

from pyro.analysis import simulation


###############################################################################
class SwitchedSystem(ContinuousDynamicSystem):
    """
    Class in construction !!!!!!
    
    """
    ############################################
    def __init__(self, n, m, p, k):
        
        super().__init__(n, m, p)
        
        
        self.k = k  # Number of discrte modes
        
    
    #############################################
    def f(self, x, u, t):
        
        u_mode = int(u[0])
        u_cont = u[1:]
        
        ################
        if u_mode == 0:
            
            dx = u_cont
            
        ################
        if u_mode == 1:
            
            dx = u_cont
        
        ################
        else:
            
            dx = np.zeros( self.n )

        return dx
    
    
    #############################
    def compute_trajectory(
        self, tf=10, n=100001):
        """ 
        Simulation of time evolution of the system
        ------------------------------------------------
        tf : final time
        n  : time steps
        """

        sim = simulation.Simulator(self, tf, n, 'euler')

        self.traj = sim.compute() # save the result in the instance

        return self.traj