# -*- coding: utf-8 -*-
"""
Created on 16/03/2020

@author: Pierre
"""

import numpy as np

from pyro.dynamic import mechanical
        
##############################################################################
        
class WCRT( mechanical.MechanicalSystem ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(3)
        
        # Name
        self.name = 'WCRT'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 1 
        self.l2  = 1
        self.l3  = 1
        self.lc1 = 1
        self.lc2 = 1
        self.lc3 = 1
        
        self.m1 = 1
        self.I1 = 0
        self.m2 = 1
        self.I2 = 0
        self.m3 = 1
        self.I3 = 0
        
        self.gravity = 9.81
        
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        
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
        c3  = np.cos( q[2] )
        s3  = np.sin( q[2] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c23,s23]
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        H = np.zeros((3,3))
        
        H[0,0] = self.m1*self.lc1**2 + self.m2*self.l1**2 + self.m3*self.l1**2
        H[1,0] = self.m3*self.l1*self.l2*s2 + self.m3*self.l1*self.lc3*c3*s2 + self.m2*self.l1*self.lc2*s2
        H[1,0] = -H[1,0]
        H[2,0] = self.m3*self.l1*self.lc3*s23
        H[2,0] = -H[2,0]
        H[0,1] = H[1,0]
        H[1,1] = self.m2*self.lc2**2 + self.m3*self.l2**2 + 2*self.m3*self.l2*self.lc3*c3 + self.m3*self.lc3**2*c3*c3
        H[2,1] = self.m3*self.lc3*self.l2*c3 + self.m3*self.lc3**2*c3*c3
        H[0,2] = H[2,0]
        H[1,2] = H[2,1]
        H[2,2] = self.m3*self.lc3**2
        
        return H
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
             
        C = np.zeros((3,3))
        
        C[0,0] = 0
        C[1,0] = 0
        C[2,0] = 0
        C[0,1] = self.m2*self.l1*self.lc2*c2*dq[1] + self.m3*self.l1*self.l2*c2*dq[1] + self.m3*self.l1*self.lc3*c2*c3*dq[1] - self.m3*self.l1*self.lc3*s2*s3*dq[2]
        C[0,1] = -C[0,1]
        C[1,1] = -self.m3*self.l2*self.lc3*s3*dq[2] - self.m3*self.lc3**2*s3*c3*dq[2]
        C[2,1] = self.m3*self.l1*self.lc3*c2*c3*dq[0] + self.m3*self.l2*self.lc3*s3*dq[1] + self.m3*self.lc3**2*s3*c3*dq[1] 
        C[0,2] = self.m3*self.l1*self.lc3*c23*dq[1] + self.m3*self.l1*self.lc3*c23*dq[2]
        C[0,2] = -C[0,2]
        C[1,2] = -C[2,1] - self.m3*self.l2*self.lc3*s3*dq[2] - 2*self.m3*self.lc3**2*s3*c3*dq[2]
        C[2,2] = 0
      
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
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        g=self.gravity
        
        G = np.zeros(3)
        
        G[0] = 0 + self.k1*q[0]
        G[1] = self.m2*g*self.lc2*c2 + self.m3*g*self.l2*c2 + self.m3*g*self.lc3*c23 + self.k2*q[1]
        G[2] = self.m3*g*self.lc3*c23 + self.k3*q[2]

        return G
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((3,3))
        
        D[0,0] = self.d1
        D[1,0] = 0
        D[2,0] = 0
        D[0,1] = 0
        D[1,1] = self.d2
        D[2,1] = 0
        D[0,2] = 0
        D[1,2] = 0
        D[2,2] = self.d3
        
        d = np.dot( D , dq )
        
        return d
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 2
        
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
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        pts[1,0] = self.l1*s1
        pts[1,1] = self.l1*c1
        pts[1,2] = 0
        
        pts[2,0] = self.l1*s1 + self.l2*c1*c2 
        pts[2,1] = self.l1*c1 - self.l2*c2*s1
        pts[2,2] = self.l2*s2
        
        pts[3,0] = self.l1*s1 + self.l2*c1*c2 -self.l3*(c1*s2*s3-c1*c2*c3)
        pts[3,1] = self.l1*c1 - self.l2*c2*s1 + self.l3*(s1*s2*s3-s1*c2*c3)
        pts[3,2] = self.l2*s2 + self.l3*(c2*s3+c3*s2)
        
        lines_pts.append( pts )
                 
        return lines_pts
        
##############################################################################
        
        
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """

    
    sys = WCRT()
    
    # No damping
    sys.d1 = 0
    sys.d2 = 0
    sys.d3 = 0
    
    sys.x0 = np.array([0.,1.6,0.,0,0,0])
    sys.animate_simulation( is_3d = True )