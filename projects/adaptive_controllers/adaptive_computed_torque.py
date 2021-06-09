#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:15:49 2020

@author: alex
"""

import numpy as np
from pyro.control import controller


##############################################################################
class SinglePendulumAdaptativeController( controller.DynamicController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    ############################
    def __init__( self , model , traj = None ):
        """ """ 
        self.name = 'Adaptive controller'

        # Params
        self.A = np.zeros(2)
        self.T=np.eye(2)
        self.Kd = 1
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        self.model=model
        
        k = model.dof   
        m = model.m
        p = model.p
        l = 2
        
        super().__init__(k, l, m, p)
        
        # Init internal states
        self.z0 = np.array([0.2,0.2])
        
        
        self.internal_state_label = []
        self.internal_state_units = []
        
        for i in range(l):
            self.internal_state_label.append('a' +str(i))
            self.internal_state_units.append('')
            
        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
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
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        Given actual state, compute torque necessarly to guarantee convergence
        """
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = self.Kd*s
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
    ############################
    def b(self, z, x, q_d, t):
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        Y = np.zeros(2)
        dz = np.zeros(2)

        Y[0]=ddq_r
        Y[1]=np.sin(q)
        b = Y * s
        dz=-1*np.dot( self.T , b )
        
        return dz
    
    ############################
    def c(self , z , x , q_d , t = 0):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )

        Y = np.zeros(2)
        Y[0]=ddq_r
        Y[1]=np.sin(q)
        
        self.A = self.get_z_integral(z)
                
        u                     = self.adaptative_torque( Y , s  , q , t )
        
        return u
    
    ############################
    def get_z_integral(self, z):
        """ get intergral error internal states """
        
        return z[:self.l]

##############################################################################
        
class DoublePendulumAdaptativeController(  controller.DynamicController ):
    """ 
    
    """
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        self.name = 'Adaptive controller'

        self.A = np.zeros(5)
        self.guess = np.zeros(5)
        self.T=np.eye(5)
        self.Kd = np.eye(2)
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        self.model=model
        
        k = model.dof   
        m = model.m
        p = model.p
        l = 5
        
        super().__init__(k, l, m, p)
        
        
        
        # Init states
        self.z0 = np.zeros(5)
        
        self.internal_state_label = []
        self.internal_state_units = []
        
        for i in range(l):
            self.internal_state_label.append('a' +str(i))
            self.internal_state_units.append('')
        
        
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

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
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
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
    
                        
    ############################
    def b(self, z, x, q_d, t):
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        Y = np.zeros((2,5))
        dz = np.zeros(5)
                
        Y[0,0]=ddq_r[0]*c2
        Y[0,1]=ddq_r[1]*c2
        Y[0,2]=s2*dq[1]*dq_r[0]
        Y[0,3]=s2*(dq[0]+dq[1])*dq_r[1]
        Y[0,4]=s1+s12
        Y[1,0]=ddq_r[0]*c2
        Y[1,1]=ddq_r[1]
        Y[1,2]=s2*dq[0]*dq_r[0]
        Y[1,3]=0
        Y[1,4]=s12
        
        b = np.dot(Y.T,s)
        dz=-1*np.dot( self.T , b )
        
        return dz
    
        
    ############################
    def c( self , z , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        Y = np.zeros((2,5))
                
        Y[0,0]=ddq_r[0]*c2
        Y[0,1]=ddq_r[1]*c2
        Y[0,2]=s2*dq[1]*dq_r[0]
        Y[0,3]=s2*(dq[0]+dq[1])*dq_r[1]
        Y[0,4]=s1+s12
        Y[1,0]=ddq_r[0]*c2
        Y[1,1]=ddq_r[1]
        Y[1,2]=s2*dq[0]*dq_r[0]
        Y[1,3]=0
        Y[1,4]=s12
        
        self.A= self.guess + self.get_z_integral(z)
                
        u                     = self.adaptative_torque(  Y , s  , q , t )
        
        return u
    
    
    ############################
    def get_z_integral(self, z):
        """ get intergral error internal states """
        
        return z[:self.l]
    
    
##############################################################################

class AdaptativeController_WCRT( controller.DynamicController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        self.name = 'Adaptive controller'
        
        # Params
        self.A = np.zeros(8)
        self.T=np.eye(8)
        self.Kd = np.eye(3)
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        self.model=model
        
        k = model.dof   
        m = model.m
        p = model.p
        l = self.A.shape[0]
        
        super().__init__(k, l, m, p) 
        
        
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

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
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
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
    
                        
    ############################
    def b(self, z, x, q_d, t):
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        Y = np.zeros((3,8))
        dz = np.zeros(8)
        
        Y[0,0]=ddq_r[0]
        Y[0,1]=ddq_r[1]*(s2+s2*c3)
        Y[0,2]=ddq_r[2]*s23
        Y[0,3]=(dq[2]*s2*s3+dq[1]*c2+dq[1]*c2*c3)*dq_r[1]
        Y[0,4]=(dq[2]+dq[1])*c23*dq_r[2]
        Y[0,5]=0
        Y[0,6]=0
        Y[0,7]=q[0]
        Y[1,0]=ddq_r[0]*(s2+s2*c3)
        Y[1,1]=ddq_r[1]*(1+c3+c3**2)
        Y[1,2]=ddq_r[2]*(c3+c3**2)
        Y[1,3]=(dq[2]*(s3+s3*c3))*dq_r[1]
        Y[1,4]=(dq[2]*(s3+s3*c3)+dq[1]*(s3+s3*c3)+dq[0]*(c2*c3))*dq_r[2]
        Y[1,5]=c2
        Y[1,6]=c23
        Y[1,7]=q[1]
        Y[2,0]=ddq_r[0]*s23
        Y[2,1]=ddq_r[1]*(c3+c3**2)
        Y[2,2]=ddq_r[2]
        Y[2,3]=(dq[1]*(s3+s3*c3)+dq[1]*c2*c3)*dq_r[1]
        Y[2,4]=0
        Y[2,5]=0
        Y[2,6]=c23
        Y[2,7]=q[2]
        
        b = np.dot(Y.T,s)
        dz=-1*np.dot( self.T , b )
            
        return dz
    
    
    ############################
    def c( self , z , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x )
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        Y = np.zeros((3,8))
                
        Y[0,0]=ddq_r[0]
        Y[0,1]=ddq_r[1]*(s2+s2*c3)
        Y[0,2]=ddq_r[2]*s23
        Y[0,3]=(dq[2]*s2*s3+dq[1]*c2+dq[1]*c2*c3)*dq_r[1]
        Y[0,4]=(dq[2]+dq[1])*c23*dq_r[2]
        Y[0,5]=0
        Y[0,6]=0
        Y[0,7]=q[0]
        Y[1,0]=ddq_r[0]*(s2+s2*c3)
        Y[1,1]=ddq_r[1]*(1+c3+c3**2)
        Y[1,2]=ddq_r[2]*(c3+c3**2)
        Y[1,3]=(dq[2]*(s3+s3*c3))*dq_r[1]
        Y[1,4]=(dq[2]*(s3+s3*c3)+dq[1]*(s3+s3*c3)+dq[0]*(c2*c3))*dq_r[2]
        Y[1,5]=c2
        Y[1,6]=c23
        Y[1,7]=q[1]
        Y[2,0]=ddq_r[0]*s23
        Y[2,1]=ddq_r[1]*(c3+c3**2)
        Y[2,2]=ddq_r[2]
        Y[2,3]=(dq[1]*(s3+s3*c3)+dq[1]*c2*c3)*dq_r[1]
        Y[2,4]=0
        Y[2,5]=0
        Y[2,6]=c23
        Y[2,7]=q[2]
        
        self.A =self.get_z_integral( z )
                         
        u                     = self.adaptative_torque( Y , s  , q , t )
        
        return u
    

    ############################
    def get_z_integral(self, z):
        """ get intergral error internal states """
        
        return z[:self.l]
    
        
##############################################################################