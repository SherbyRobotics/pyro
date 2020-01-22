# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np

import sys
if sys.version > '3':
    from abc import ABC
else:
    from abc import ABCMeta

from collections import namedtuple

from copy import copy

from scipy.integrate import cumtrapz

from . import Trajectory


##########################################################################
# Cost functions
##########################################################################

if sys.version > '3':
    class CostFunction(ABC):
        """
        Mother class for cost functions of continuous dynamical systems
        ----------------------------------------------
        n : number of states
        m : number of control inputs
        p : number of outputs
        ---------------------------------------
        J = int( g(x,u,y,t) * dt ) + h( x(T) , y(T) , T )

        """

        ###########################################################################
        # The two following functions needs to be implemented by child classes
        ###########################################################################

        ############################
        def __init__(self):
            self.INF = 1E3
            self.EPS = 1E-3

        #############################
        def h(self, x, t=0):
            """ Final cost function """

            raise NotImplementedError

        #############################
        def g(self, x, u, y, t):
            """ step cost function """

            raise NotImplementedError

        def eval(self, traj):
            """Compute cost of simulation

            Parameters
            ----------
            traj : instance of `pyro.analysis.Trajectory`

            Returns
            -------
            A new instance of the input trajectory, with updated `J` and `dJ` fields

            J : array of size ``traj.n`` (number of timesteps in trajectory)
                Cumulative value of cost integral at each time step. The total cost is
                therefore ``J[-1]``.

            dJ : array of size ``traj.n`` (number of timesteps in trajectory)
                Value of cost function evaluated at each point of the tracjectory.
            """

            dJ = np.empty(traj.n)
            for i in range(traj.n):
                x = traj.x[i, :]
                u = traj.u[i, :]
                y = traj.y[i, :]
                t = traj.t[i]
                dJ[i] = self.g(x, u, y, t)

            J = cumtrapz(y=dJ, x=traj.t, initial=0)

            new_traj = copy(traj)
            new_traj.J = J
            new_traj.dJ = dJ

            return new_traj
else:
    class CostFunction():
        __metaclass__ = ABCMeta
        """ 
        Mother class for cost functions of continuous dynamical systems
        ----------------------------------------------
        n : number of states
        m : number of control inputs
        p : number of outputs
        ---------------------------------------
        J = int( g(x,u,y,t) * dt ) + h( x(T) , y(T) , T )

        """

        ###########################################################################
        # The two following functions needs to be implemented by child classes
        ###########################################################################

        ############################
        def __init__(self):
            self.INF = 1E3
            self.EPS = 1E-3

        #############################
        def h(self, x, t=0):
            """ Final cost function """

            raise NotImplementedError

        #############################
        def g(self, x, u, y, t):
            """ step cost function """

            raise NotImplementedError

        def eval(self, traj):
            """Compute cost of simulation

            Parameters
            ----------
            traj : instance of `pyro.analysis.Trajectory`

            Returns
            -------
            A new instance of the input trajectory, with updated `J` and `dJ` fields

            J : array of size ``traj.n`` (number of timesteps in trajectory)
                Cumulative value of cost integral at each time step. The total cost is
                therefore ``J[-1]``.

            dJ : array of size ``traj.n`` (number of timesteps in trajectory)
                Value of cost function evaluated at each point of the tracjectory.
            """

            dJ = np.empty(traj.n)
            for i in range(traj.n):
                x = traj.x[i, :]
                u = traj.u[i, :]
                y = traj.y[i, :]
                t = traj.t[i]
                dJ[i] = self.g(x, u, y, t)

            J = cumtrapz(y=dJ, x=traj.t, initial=0)

            new_traj = copy(traj)
            new_traj.J = J
            new_traj.dJ = dJ

            return new_traj

#############################################################################
     
class QuadraticCostFunction( CostFunction ):
    """ 
    Quadratic cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    p : number of outputs
    ---------------------------------------
    J = int( g(x,u,y,t) * dt ) + h( x(T) , y(T) , T )
    
    g = xQx + uRu + yVy
    h = 0
    
    """
    
    ############################
    def __init__(self, q, r, v):

        if sys.version > '3':
            super().__init__()
        else:
            super(QuadraticCostFunction, self).__init__()

        self.n = q.shape[0]
        self.m = r.shape[0]
        self.p = v.shape[0]

        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)
        self.ybar = np.zeros(self.p)

        # Quadratic cost weights
        self.Q = np.diag( q )
        self.R = np.diag( r )
        self.V = np.diag( v )

        self.ontarget_check = True

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        return 0
    
    
    #############################
    def g(self, x, u, y, t):
        """ Quadratic additive cost """

        # Check dimensions
        if not x.shape[0] == self.Q.shape[0]:
            raise ValueError(
                "Array x of shape %s does not match weights Q with %d components" \
                % (x.shape, self.Q.shape[0])
            )
        if not u.shape[0] == self.R.shape[0]:
            raise ValueError(
                "Array u of shape %s does not match weights R with %d components" \
                % (u.shape, self.R.shape[0])
            )
        if not y.shape[0] == self.V.shape[0]:
            raise ValueError(
                "Array y of shape %s does not match weights V with %d components" \
                % (y.shape, self.V.shape[0])
            )

        dx = x - self.xbar
        du = u - self.ubar
        dy = y - self.ybar
        
        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot( du.T , np.dot(  self.R , du ) ) +
               np.dot( dy.T , np.dot(  self.V , dy ) ) )
        
        # set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
        
        return dJ
    

##############################################################################

class TimeCostFunction( CostFunction ):
    """ 
    Mother class for cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    p : number of outputs
    ---------------------------------------
    J = int( g(x,u,y,t) * dt ) + h( x(T) , y(T) , T ) = T
    
    g = 1
    h = 0
    
    """
    
    ############################
    def __init__(self, xbar ):

        if sys.version > '3':
            super().__init__()
        else:
            super(TimeCostFunction, self).__init__()
        
        self.xbar = xbar
        
        self.ontarget_check = True
        
    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        return 0
    
    
    #############################
    def g(self, x , u , y, t = 0 ):
        """ Unity """

        if (x.shape[0] != self.xbar.shape[0]):
            raise ValueError("Got x with %d values, but xbar has %d values" %
                             (x.shape[1], self.xbar.shape[0]))

        dJ = 1
        
        if self.ontarget_check:
            dx = x - self.xbar
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
                
        return dJ

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass