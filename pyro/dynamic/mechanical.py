# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:45:37 2018

@author: Alexandre
"""


###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import system
###############################################################################




###############################################################################
        
class MechanicalSystem( system.ContinuousDynamicSystem ):
    """ 
    Mechanical system with Equation of Motion in the form of
    -------------------------------------------------------
    H(q) ddq + C(q,dq) dq + d(q,dq) + g(q) = B(q) u
    -------------------------------------------------------
    q      :  dim = (dof, 1)   : position variables 
    dq     :  dim = (dof, 1)   : velocity variables     
    ddq    :  dim = (dof, 1)   : acceleration variables
    u      :  dim = (m, 1)     : force input variables
    H(q)   :  dim = (dof, dof) : inertia matrix
    C(q)   :  dim = (dof, dof) : corriolis matrix
    B(q)   :  dim = (dof, m)   : actuator matrix
    ddq    :  dim = (dof, 1)   : acceleration variables
    d(q,dq):  dim = (dof, 1)   : state-dependent dissipative forces
    g(q)   :  dim = (dof, 1)   : state-dependent conservatives forces
    
    """
    
    ############################
    def __init__(self, dof = 1 ):
        """ """
        
        # Degree of Freedom
        self.dof = dof
        
        # Dimensions
        n = dof * 2 
        m = dof  
        p = dof * 2
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, n, m, p)
        
        # Name
        self.name = str(dof) + 'DoF Mechanical System'
        
        # Labels, bounds and units
        for i in range(dof):
            # joint angle states
            self.x_ub[i] = + np.pi * 2
            self.x_lb[i] = - np.pi * 2
            self.state_label[i] = 'Angle '+ str(i)
            self.state_units[i] = '[rad]'
            # joint velocity states
            self.x_ub[i+dof] = + np.pi * 2
            self.x_lb[i+dof] = - np.pi * 2
            self.state_label[i+dof] = 'Velocity ' + str(i)
            self.state_units[i+dof] = '[rad/sec]'
        for i in range(dof):
            self.u_ub[i] = + 5
            self.u_lb[i] = - 5
            self.input_label[i] = 'Torque ' + str(i)
            self.input_units[i] ='[Nm]'
        self.output_label = self.state_label
        self.output_units = self.state_units
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.diag( np.ones( self.dof ) ) # Default is identity matrix
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros( ( self.dof , self.dof ) ) # Default is zeros matrix
        
        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) # Default is identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) # Default is zero vector
        
        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(self.dof ) # Default is zero vector
        
        return d
    
    
    ###########################################################################
    # No need to overwrite the following functions for custom system
    ###########################################################################
    
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        : self.dof ]
        dq = x[ self.dof : self.n   ]
        
        return [ q , dq ]
        
        
    #############################
    def q2x( self, q , dq ):
        """ from angle and speeds (q,dq) to state vector (x) """
        
        x = np.zeros( self.n )
        
        x[ 0        : self.dof ] = q
        x[ self.dof : self.n   ] = dq
        
        return x
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute configuration variables """
        
        return self.x2q(x)[0]
    
    
    ##############################
    def generalized_forces(self, q  , dq  , ddq , t = 0 ):
        """ Computed generalized forces given a trajectory """  
        
        H = self.H( q )
        C = self.C( q , dq )
        g = self.g( q )
        d = self.d( q , dq )
                
        # Generalized forces
        forces = np.dot( H , ddq ) + np.dot( C , dq ) + g + d
        
        return forces
    
    
    ##############################
    def actuator_forces(self, q  , dq  , ddq , t = 0 ):
        """ Computed actuator forces given a trajectory (inverse dynamic) """  
        
        B = self.B( q )
                
        # Generalized forces
        forces = self.generalized_forces( q , dq , ddq , t )
        
        # Actuator forces
        u = np.dot( np.linalg.inv( B ) , forces )
        
        return u
    
    
    ##############################
    def ddq(self, q , dq , u , t = 0 ):
        """ Computed accelerations given actuator forces (foward dynamic) """  
        
        H = self.H( q )
        C = self.C( q , dq )
        g = self.g( q  )
        d = self.d( q , dq)
        B = self.B( q )
        
        ddq = np.dot( np.linalg.inv( H ) ,  ( np.dot( B , u )  
                                            - np.dot( C , dq ) - g - d ) )
        
        return ddq
    
    
    ###########################################################################
    def f(self, x , u , t = 0 ):
        """ 
        Continuous time foward dynamics evaluation
        
        dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        
        # from state vector (x) to angle and speeds (q,dq)
        [ q , dq ] = self.x2q( x )       
        
        # compute joint acceleration 
        ddq = self.ddq( q , dq , u , t ) 
        
        # from angle and speeds diff (dq,ddq) to state vector diff (dx)
        dx = self.q2x( dq , ddq )        
        
        return dx
    
    
    ###########################################################################
    def kinetic_energy(self, q  , dq ):
        """ Compute kinetic energy of manipulator """  
        
        e_k = 0.5 * np.dot( dq , np.dot( self.H( q ) , dq ) )
        
        return e_k
        
        
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = MechanicalSystem( 2 )
    
    sys.show(  q = np.array([ 1.0, 2.0]) )
    sys.show3( q = np.array([-0.5, 1.5]) )
    
    sys.ubar = np.array([1,2])
    sys.x0   = np.array([0,0,0,0])
    
    sys.plot_trajectory()
    sys.animate_simulation()
        