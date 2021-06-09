# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 20:13:30 2019

@author: alxgr
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import cartpole
from pyro.control import controller
###############################################################################

class CartPoleController( controller.StaticController ) :
    
    ############################
    def __init__( self , sys_model ):
        """ """
        
        # Dimensions
        self.dof = 2
        self.k   = 1 
        self.m   = 1
        self.p   = 4 
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Cart Pole Controller'
        
        # Gains
        self.kd = 5
        self.kp = 3
        
        self.rbar = np.zeros(1)
        
        
        # Model
        self.sys = sys_model

    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # State feedback
        q  = y[ 0        :     self.dof   ]
        dq = y[ self.dof : 2 * self.dof   ]
        
        H = self.sys.H(q)
        C = self.sys.C(q,dq)
        d = self.sys.d(q,dq)
        g = self.sys.g(q)
        
        phi = np.dot(C,dq) + d + g
        
        H11 = H[0,0]
        H12 = H[0,1]
        H22 = H[1,1]
        
        A = H12 - H11 * H22 / H12
        B = phi[0] - H11 / H12 * phi[1]
        
        # Energy
        T = 0.5 * H22 * dq[1]**2
        V = self.sys.m2 * self.sys.gravity * self.sys.l2 * np.cos( q[1] ) 
        E = T + V
        Ed = self.sys.m2 * self.sys.gravity * self.sys.l2
        
        Ee = E-Ed
        
        #Energy shaping swing up
        if ( np.abs(H12) > 0.01 ) & ( dq[1] > 0.1 ):
            
            mgl = self.sys.m2 * self.sys.gravity * self.sys.l2 * np.sin( q[1])
            
            gain_e = 5
            
            ddq_r = (mgl - Ee * gain_e / dq[1] ) / H22
            
            damping_base = 5
            
            u[0] = A * ddq_r + B - damping_base * dq[0]
            
            
        # On target stabilization
        if (q[1]**2 < 0.2 ) & ( dq[1]**2 < 0.6 ):
            
            u[0] = +100 * q[1] + 22 * dq[1] + 1 * q[0]  + 2 * dq[0]
            
        
        # Saturation
        umax = 50
        
        if u[0] > umax:
            
            u[0] = umax
        elif u[0] < -umax:
            u[0] = -umax

        
        return u



###############################################################################

sys = cartpole.UnderActuatedRotatingCartPole()

sys.d1 = 0
sys.d2 = 0

ctl = CartPoleController( sys )

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([0,-3,0,-1])
cl_sys.compute_trajectory(10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation(time_factor_video=1.0, is_3d=True)