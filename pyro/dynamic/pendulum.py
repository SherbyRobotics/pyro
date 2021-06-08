# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:07:39 2018

@author: nvidia
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import mechanical
###############################################################################


###############################################################################

class SinglePendulum( mechanical.MechanicalSystem ):
    """Pendulum with a point mass and inertial rod.

    Attributes
    ----------
    l1 : float
        Length of pendulum rod. Only used for display.
    lc1 : float
        Distance of point mass to pivot.
    m1 : float
        Mass value of point mass.
    I1 : float
        Moment of inertia of pendulum rod (without mass) about the pendulum pivot. The
        total inertia of the pendulum is calculated as
        ``I_pendulum = I1 + (m1 * lc1**2)``.
    gravity : float
        Constant of gravitational acceleration
    d1: float
        Damping force factor
    """

    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(1)
        
        # Name
        self.name = 'Single Pendulum'
        
        # params
        self.setparams()
        
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        # kinematic
        self.l1  = 2 
        self.lc1 = 1
        
        # dynamic
        self.m1       = 1
        self.I1       = 1
        self.gravity  = 9.81
        self.d1       = 0
        
        
    ##############################
    def trig(self, q ):
        """ Compute cos and sin """
        
        c1  = np.cos( q )
        s1  = np.sin( q )

        return [c1,s1]
    
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((self.dof,self.dof))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((self.dof,self.dof))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) 
        
        [c1,s1] = self.trig( q )
        
        g[0] = self.m1 * self.gravity * self.lc1 * s1

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d    = np.zeros( self.dof ) 
        
        d[0] = self.d1 * dq[0]
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 5
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        # pendulum
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1] = self.trig( q )
        
        pts[1,0] =   s1 * self.l1
        pts[1,1] = - c1 * self.l1
        
        lines_pts.append( pts )
                
        return lines_pts
        
        
        
##############################################################################
        
class DoublePendulum( mechanical.MechanicalSystem ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(2)
        
        # Name
        self.name = 'Double Pendulum'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 1 
        self.l2  = 1
        self.lc1 = 1
        self.lc2 = 1
        
        self.m1 = 1
        self.I1 = 0
        self.m2 = 1
        self.I2 = 0
        
        self.gravity = 9.81
        
        self.d1 = 0
        self.d2 = 0
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        
        return [c1,s1,c2,s2,c12,s12]
        
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        H = np.zeros((2,2))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1 + self.m2 * ( self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2 ) + self.I2
        H[1,0] = self.m2 * self.lc2**2 + self.m2 * self.l1 * self.lc2 * c2 + self.I2
        H[0,1] = H[1,0]
        H[1,1] = self.m2 * self.lc2 ** 2 + self.I2
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        h = self.m2 * self.l1 * self.lc2 * s2
        
        C = np.zeros((2,2))
        
        C[0,0] = - h  * dq[1]
        C[1,0] =   h  * dq[0]
        C[0,1] = - h * ( dq[0] + dq[1] )
        C[1,1] = 0

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1 ) * self.gravity
        g2 = self.m2 * self.lc2 * self.gravity
        
        G = np.zeros(2)
        
        G[0] = - g1 * s1 - g2 * s12
        G[1] = - g2 * s12

        return G
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((2,2))
        
        D[0,0] = self.d1
        D[1,0] = 0
        D[0,1] = 0
        D[1,1] = self.d2
        
        d = np.dot( D , dq )
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 3
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        ###########################
        # pendulum kinematic
        ###########################
        
        pts      = np.zeros(( 3 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        pts[1,0] = self.l1 * s1
        pts[1,1] = self.l1 * c1
        
        pts[2,0] = self.l1 * s1 + self.l2 * s12
        pts[2,1] = self.l1 * c1 + self.l2 * c12
        
        lines_pts.append( pts )
                
        return lines_pts
    
    
    
###############################################################################

class TwoIndependentSinglePendulum( mechanical.MechanicalSystem ):
    """ Two Pendulum with a point mass and inertial rod.

    """

    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(2)
        
        # Name
        self.name = 'Two Independent Single Pendulum'
        
        # params
        
        # kinematic
        self.l1  = 2 
        self.lc1 = 1
        
        # dynamic
        self.m1       = 1
        self.I1       = 1
        self.gravity  = 9.81
        self.d1       = 0
        
        
    ##############################
    def trig(self, q ):
        """ Compute cos and sin """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )

        return [c1,s1,c2,s2]
    
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((self.dof,self.dof))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1
        H[1,1] = self.m1 * self.lc1**2 + self.I1
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((self.dof,self.dof))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) 
        
        [c1,s1,c2,s2] = self.trig( q )
        
        g[0] = self.m1 * self.gravity * self.lc1 * s1
        g[1] = self.m1 * self.gravity * self.lc1 * s2

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d    = np.zeros( self.dof ) 
        
        d[0] = self.d1 * dq[0]
        d[1] = self.d1 * dq[1]
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 5
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        offset = 2
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        # pendulum no 1
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([- offset,0,0])
        
        [c1,s1,c2,s2] = self.trig( q )
        
        pts[1,:] = np.array([ s1 * self.l1 - offset , - c1 * self.l1 ,0])
        
        lines_pts.append( pts )
        
        # pendulum no 1
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([offset,0,0])
        
        [c1,s1,c2,s2] = self.trig( q )
        
        pts[1,:] = np.array([ s2 * self.l1 + offset , - c2 * self.l1 ,0])
        
        lines_pts.append( pts )
                
        return lines_pts
        
        
        
        
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = DoublePendulum()
    
    #sys = TwoIndependentSinglePendulum()
    
    sys.x0 = np.array([0.1,0.9,0,0])
    tf = 20
    sys.compute_trajectory( tf )
    sys.plot_trajectory()
    sys.animate_simulation()