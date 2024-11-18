# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

###############################################################################
import numpy as np
from copy import copy
from scipy.integrate import cumulative_trapezoid
###############################################################################


###############################################################################
# Mother cost function class
###############################################################################

class CostFunction():
    """
    Mother class for cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T )

    """

    ############################
    def __init__(self):
        self.INF = 1E3
        self.EPS = 1E-3
        
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################

    #############################
    def h(self, x, t ):
        """ Final cost function """

        raise NotImplementedError

    #############################
    def g(self, x, u, t ):
        """ step cost function """

        raise NotImplementedError
        
    ###########################################################################
    # Method using h and g
    ###########################################################################
        
    #############################
    def trajectory_evaluation(self, traj):
        """
        
        Compute cost and add it to simulation data

        Parameters
        ----------
        traj : instance of `pyro.analysis.Trajectory`

        Returns
        -------
        new_traj : A new instance of the input trajectory, with updated `J` and
        `dJ` fields

        J : array of size ``traj.time_steps`` (number of timesteps in 
        trajectory)
            Cumulative value of cost integral at each time step. The total 
            cost is
            therefore ``J[-1]``.

        dJ : array of size ``traj.time_steps`` (number of timesteps in 
        trajectory)
            Value of cost function evaluated at each point of the tracjectory.
        """

        dJ = np.empty(traj.time_steps)
        for i in range(traj.time_steps):
            x = traj.x[i, :]
            u = traj.u[i, :]
            t = traj.t[i]
            dJ[i] = self.g( x, u, t)

        J = cumulative_trapezoid(y=dJ, x=traj.t, initial=0)

        new_traj = copy(traj)
        new_traj.J = J
        new_traj.dJ = dJ

        return new_traj
    
    
###############################################################################
# Basic cost functions
###############################################################################
     
class QuadraticCostFunction( CostFunction ):
    """ 
    Quadratic cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T )
    
    g = xQx + uRu 
    h = xSx
    
    """
    
    ############################
    def __init__(self, n, m):
        
        CostFunction.__init__(self)
        
        # dimensions
        self.n = n
        self.m = m
        
        # nominal values
        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)

        # Quadratic cost weights
        self.Q = np.diag( np.ones(n)  )
        self.R = np.diag( np.ones(m)  )
        self.S = np.diag( np.zeros(n) )
        
        # Optionnal zone of zero cost if ||x - xbar || < EPS 
        self.ontarget_check = True
        
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m )
        
        instance.xbar = sys.xbar
        instance.ubar = sys.ubar
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        # Delta values with respect to nominal values
        dx = x - self.xbar
        
        # Quadratic terminal cost
        J_f = np.dot( dx.T , np.dot(  self.S , dx ) )
                     
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                J_f = 0
        
        return J_f
    
    
    #############################
    def g(self, x, u, t):
        """ Quadratic additive cost """
        
        """
        TODO: Add check in init
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
        """
            
        # Delta values with respect to nominal values
        dx = x - self.xbar
        du = u - self.ubar
        
        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot( du.T , np.dot(  self.R , du ) ) )
        
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
        
        return dJ
    
    
#############################################################################
     
class QuadraticCostFunctionVectorized( CostFunction ):
    """ 
    Vectorized: (x, u , t) can be trajectory of time matrices
    
    Quadratic cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T )
    
    g = xQx + uRu 
    h = xSx
    
    """
    
    ############################
    def __init__(self, n, m):
        
        CostFunction.__init__(self)
        
        # dimensions
        self.n = n
        self.m = m

        # Quadratic cost weights
        self.Q = np.diag( np.ones(n)  )
        self.R = np.diag( np.ones(m)  )
        self.S = np.diag( np.zeros(n) )
        
        self.is_vectorized = True
        
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m )
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        if x.ndim == 1 :
            
            J_f = x.T @ self.S @ x 
            
            
        else:
            
            # Quadratic terminal cost
            J_f = np.diag( x.T @ self.S @ x )
        
        return J_f
    
    
    #############################
    def g(self, x, u, t):
        """ Quadratic additive cost """
        
        if x.ndim == 1 :
            
            dJ = x.T @ self.Q @ x + u.T @ self.R @ u
            
        else:
        
            dJ = np.diag( x.T @ self.Q @ x ) + np.diag( u.T @ self.R @ u )
        
        return dJ

    

##############################################################################

class TimeCostFunction( CostFunction ):
    """ 
    Mother class for cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T ) = T
    
    g = 1
    h = 0
    
    """
    
    ############################
    def __init__(self, xbar ):

        CostFunction.__init__(self)
        
        self.xbar = xbar
        
        self.ontarget_check = True
        
    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        return 0
    
    
    #############################
    def g(self, x , u , t = 0 ):
        """ Unity """
        
        """
        if (x.shape[0] != self.xbar.shape[0]):
            raise ValueError("Got x with %d values, but xbar has %d values" %
                             (x.shape[1], self.xbar.shape[0]))
        """

        dJ = 1
        
        if self.ontarget_check:
            dx = x - self.xbar
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
                
        return dJ
    


##############################################################################
class QuadraticCostFunctionWithDomainCheck( CostFunction ):
    """ 
    Quadratic cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T )
    
    g = xQx + uRu  if x and u are allowable state and actions
    h = 0          if x and u are allowable state and actions
    
    """
    
    ############################
    def __init__(self, n, m, isavalidstate ):
        
        QuadraticCostFunction.__init__(self, n , m )
        
        self.isavalidstate = isavalidstate
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m , sys.isavalidstate )
        
        instance.xbar = sys.xbar
        instance.ubar = sys.ubar
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        # Delta values with respect to nominal values
        dx = x - self.xbar
        
        # Quadratic terminal cost
        J_f = np.dot( dx.T , np.dot(  self.S , dx ) )
                
        # Set cost to INF if not an allowable state
        if not self.isavalidstate( x ):
            J_f = self.INF
            
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                J_f = 0
        
        return J_f
    
    
    #############################
    def g(self, x, u, t):
        """ Quadratic additive cost """
            
        # Delta values with respect to nominal values
        dx = x - self.xbar
        du = u - self.ubar
        
        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot( du.T , np.dot(  self.R , du ) ) )
                
        # Set cost to INF if not an allowable state
        if not self.isavalidstate( x ):
            dJ = self.INF
            
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
        
        return dJ
    
    
    
##############################################################################

class Reachability( CostFunction ):
    
    ############################
    def __init__(self, isavalidestate , xbar = None , isontarget = None ):

        CostFunction.__init__(self)
        
        self.INF = 1E4
        self.EPS = 0.2
        
        self.isavalidestate = isavalidestate
        
        if isontarget == None:
            # default on target test is a quadratic norm check with xbar
            self.isontarget = self.norm_test 
            self.xbar       = xbar
            
        else:
            # Custom function returning bool from state
            self.isontarget = isontarget 
            
        
    #############################
    def norm_test(self, x , t = 0):
        """ Final cost function with zero value """
        
        dx = x - self.xbar
        
        isontarget = np.linalg.norm( dx ) < self.EPS
        
        return isontarget
    
        
    #############################
    def h(self, x , t = 0):
        """ Final cost """
        
        if self.isontarget( x , t ):
            
            J_f = 0 # Finishing in the target set is very good
            
        else:
            
            J_f = self.INF # Finishing not in the target set is very bad
        
        return J_f
    
    
    #############################
    def g(self, x , u , t = 0 ):
        """ Unity """
        
        if self.isavalidestate( x ):
            
            g = 0 
            
        else:
            
            g = self.INF # The system went on of bounds
        
        return g
    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass