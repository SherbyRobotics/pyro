# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:46:14 2018

@author: alxgr
"""

###############################################################################
import numpy as np
from scipy.interpolate import interp1d
###############################################################################
from pyro.control import controller
from pyro.dynamic import mechanical
###############################################################################




###############################################################################
# Computed Torque
###############################################################################
        
class ComputedTorqueController( controller.StaticController ) :
    """ 
    Inverse dynamic controller for mechanical system

    """    
    
    ############################
    def __init__(self, model = mechanical.MechanicalSystem() , traj = None ):
        """
        
        ---------------------------------------
        r  : reference signal vector  k x 1
        y  : sensor signal vector     p x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        ---------------------------------------
        u = c( y , r , t )
        
        """
        
        self.model = model
        
        # Dimensions
        self.k = model.dof   
        self.m = model.m
        self.p = model.p
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Computed Torque Controller'
        
        # Params
        self.w0   = 1
        self.zeta = 0.7 
        
        # Mode
        if traj == None:
            self.c = self.c_fixed_goal
        else:
            self.load_trajectory( traj )
            self.mode = 'interpol'
            self.c = self.c_trajectory_following
        
    
    #############################
    def c_fixed_goal( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
                
        x   = y 
        q_d = r
        
        u = self.fixed_goal_ctl( x , q_d , t )
        
        return u
    
        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        [ q , dq ]     = self.model.x2q( x )  
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )

        ddq_r          = self.compute_ddq_r( ddq_d , dq_d , q_d , dq , q )
        
        u              = self.model.actuator_forces( q , dq , ddq_r )
        
        return u
        
        
    ############################
    def compute_ddq_r( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state, compute ddq_r
        
        """
        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        ddq_r = ddq_d - 2 * self.zeta * self.w0 * dq_e - self.w0 ** 2 * q_e
        
        return ddq_r
    
        
    ############################
    def load_trajectory( self , traj  ):
        """ 
        
        Load Open-Loop trajectory solution to use as reference trajectory
        
        """
        
        self.trajectory = traj
        
        q   = traj.x[ :,    0           :     self.model.dof ]
        dq  = traj.x[ :, self.model.dof : 2 * self.model.dof ]
        ddq = traj.dx[:, self.model.dof : 2 * self.model.dof ]
        t   = traj.t
        
        # Create interpol functions
        self.q   = interp1d(t,q.T)
        self.dq  = interp1d(t,dq.T)
        self.ddq = interp1d(t,ddq.T)
        
        
    ############################
    def get_traj( self , t  ):
        """ 
        
        Find closest point on the trajectory
        
        """
        
        if t < self.trajectory.time_final :

            # Load trajectory
            q     = self.q(   t )
            dq    = self.dq(  t )
            ddq   = self.ddq( t )          

        else:
            
            q     = self.rbar
            dq    = np.zeros( self.model.dof )
            ddq   = np.zeros( self.model.dof )
        
        return ddq , dq , q
    
    
    ############################
    def traj_following_ctl( self , x , t = 0 ):
        """ 
        
        Given desired loaded trajectory and actual state, compute torques
        
        """
        
        [ q , dq ]         = self.model.x2q( x ) 
        
        ddq_d , dq_d , q_d = self.get_traj( t )

        ddq_r              = self.compute_ddq_r( ddq_d , dq_d , q_d , dq , q )
        
        u                  = self.model.actuator_forces( q , dq , ddq_r )
        
        return u
        
        
    #############################
    def c_trajectory_following( self , y , r , t ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        x = y 
        
        u = self.traj_following_ctl( x , t )
        
        
        return u
    


##############################################################################
        
class SlidingModeController( ComputedTorqueController ):
    """ 
    Sliding Mode Controller for fully actuated mechanical systems
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        super().__init__( model , traj )
        
        # Params
        self.lam  = 1   # Sliding surface slope
        self.gain = 1   # Discontinuous gain
        self.nab  = 0.1 # Min convergence rate
        
        
    ############################
    def compute_sliding_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def K( self , q , t ):
        """ Discontinuous gain matrix """
        
        dist_max = np.diag( np.ones( self.model.dof ) ) * self.gain
        conv_min = np.diag( np.ones( self.model.dof ) ) * self.nab
        
        K = dist_max + np.dot( self.model.H( q ) , conv_min ) 
        
        return K
        
        
    ############################
    def sliding_torque( self , ddq_r , s , dq , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = self.model.actuator_forces( q , dq , ddq_r )
        
        u_discontinuous = np.dot( self.K( q , t ) ,  np.sign( s ) )
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
        
        
    ############################
    def traj_following_ctl( self , x , t = 0 ):
        """ 
        
        Given desired loaded trajectory and actual state, compute torques
        
        """
        
        [ q , dq ]            = self.model.x2q( x ) 
        
        ddq_d , dq_d , q_d    = self.get_traj( t )

        [ s , dq_r , ddq_r ]  = self.compute_sliding_variables( ddq_d , dq_d , 
                                                                q_d , dq , q )
        
        u                     = self.sliding_torque( ddq_r , s , dq , q , t )
        
        return u
        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )

        [ s , dq_r , ddq_r ]  = self.compute_sliding_variables( ddq_d , dq_d , 
                                                                q_d , dq , q )
        
        u                     = self.sliding_torque( ddq_r , s , dq , q , t )
        
        return u

    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic import pendulum


    sys  = pendulum.DoublePendulum()
    ctl  = ComputedTorqueController( sys )
    
    # New cl-dynamic
    cl_sys = ctl + sys
    
    cl_sys.x0 = np.array([2,1,0,0])
    cl_sys.compute_trajectory()
    cl_sys.plot_phase_plane_trajectory(0,2)
    cl_sys.animate_simulation()
        
