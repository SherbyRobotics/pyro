#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 05:43:08 2021

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
from pyro.kinematic import geometry
from pyro.kinematic import drawing
###############################################################################


##############################################################################
# 2D planar drone
##############################################################################
        
class Drone2D( mechanical.MechanicalSystem ):
    """ 
    Equations of Motion
    -------------------------
    
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        dof       = 3   
        actuators = 2
        
        # initialize standard params
        mechanical.MechanicalSystem.__init__( self, dof, actuators )
        
        # Labels
        self.name = '2D drone model'
        self.state_label = ['x','y','theta','vx','vy','w']
        self.input_label = ['T1', 'T2']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_units = ['[N]', '[N]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+5,+2,+1,10,10,10])
        self.x_lb = np.array([-5,-2,-1,-10,-10,-10])
        
        # Model param
        self.mass           = 1
        self.inertia        = 0.1
        self.truster_offset = 0.5
        self.gravity        = 9.81
        self.cda            = 0.1
        
        # Kinematic param
        self.width  = 1
        self.height = 0.2
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((3,3))
        
        H[0,0] = self.mass
        H[1,1] = self.mass
        H[2,2] = self.inertia
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((3,3))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,2))
        
        # TODO PLACE HOLDER
        B[0,0] = -np.sin( q[2] )
        B[0,1] = -np.sin( q[2] )
        B[1,0] = np.cos( q[2] )
        B[1,1] = np.cos( q[2] )
        B[2,0] = -self.truster_offset
        B[2,1] = self.truster_offset
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros(3)
        
        g[1] =  self.mass * self.gravity

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(3)
        
        d[0] = dq[0]*abs(dq[0]) * self.cda + dq[0] * 0.01
        d[1] = dq[1]*abs(dq[1]) * self.cda + dq[1] * 0.01
        d[2] = dq[2]*abs(dq[2]) * 0        + dq[2] * 0.01
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.width * 3
        
        x = q[0]
        y = q[1]
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
                
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
        lines_style = []
        lines_color = []
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--')
        lines_color.append( 'k' )
        
        ###########################
        # drone body
        ###########################
        
        x = q[0]
        y = q[1]
        s = np.sin(q[2])
        c = np.cos(q[2])
        l = self.width
        h = self.height
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([x+l*c-h*s,y+l*s+h*c,0])
        pts[1,:] = np.array([x+l*c,y+l*s,0])
        pts[2,:] = np.array([x-l*c,y-l*s,0])
        pts[3,:] = np.array([x-l*c-h*s,y-l*s+h*c,0])
        
        
        lines_pts.append( pts )
        lines_style.append( 'o-')
        lines_color.append( 'b' )
        
        ###########################
        # drone cg
        ###########################
        
        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([x,y,0])
        
        lines_pts.append( pts )
        lines_style.append( 'o')
        lines_color.append( 'b' )
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show trust vectors
        
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        q, dq = self.x2q(x)
        
        x     = q[0]
        y     = q[1]
        theta = q[2]
        
        world_T_body = geometry.transformation_matrix_2D( theta , x , y )
        
        ###########################
        # drone trust force vectors
        ###########################

        ###########################
        # Trust vector
        ###########################
        
        # Max trust --> arrow is long as airplane
        f_scale = self.width / (self.u_ub[0] - self.u_lb[0])
        
        vy = u[0] * f_scale

        bx = self.width * -1.0
        by = self.height * 1.0
        
        trust_arrow_body = drawing.arrow_from_length_angle( vy , np.pi * 0.5 , bx , by )
        
        trust_arrow_world = drawing.transform_points_2D( world_T_body , trust_arrow_body )
        
        lines_pts.append( trust_arrow_world )
        lines_style.append( '-')
        lines_color.append( 'r')

        vy = u[1] * f_scale

        bx = self.width * +1.0
        by = self.height * 1.0
        
        trust_arrow_body = drawing.arrow_from_length_angle( vy , np.pi * 0.5 , bx , by )
        
        trust_arrow_world = drawing.transform_points_2D( world_T_body , trust_arrow_body )
        
        lines_pts.append( trust_arrow_world )
        lines_style.append( '-')
        lines_color.append( 'r')
        
        # xcg = x[0]
        # ycg = x[1]
        # s = np.sin(x[2])
        # c = np.cos(x[2])
        # l = self.width
        # h = self.height 
        # h2 = self.height * u[0]
        
        # pts      = np.zeros(( 5 , 3 ))
        # pts[0,:] = np.array([xcg+l*c-h*s,ycg+l*s+h*c,0])
        # pts[1,:] = np.array([xcg+l*c-h2*s,ycg+l*s+h2*c,0])
        # pts[2,:] = pts[1,:] + np.array([-h*c+h*s,-h*s-h*c,0])
        # pts[3,:] = pts[1,:] 
        # pts[4,:] = pts[1,:] + np.array([h*c+h*s,h*s-h*c,0])
        
        # lines_pts.append( pts )
        # lines_style.append( '-')
        # lines_color.append( 'r' )
        
        # pts      = np.zeros(( 5 , 3 ))
        # pts[0,:] = np.array([xcg-l*c-h*s,ycg-l*s+h*c,0])
        # pts[1,:] = np.array([xcg-l*c-h2*s,ycg-l*s+h2*c,0])
        # pts[2,:] = pts[1,:] + np.array([-h*c+h*s,-h*s-h*c,0])
        # pts[3,:] = pts[1,:] 
        # pts[4,:] = pts[1,:] + np.array([h*c+h*s,h*s-h*c,0])
        
        # lines_pts.append( pts )
        # lines_style.append( '-')
        # lines_color.append( 'r' )
                
        return lines_pts , lines_style , lines_color
    
  

###############################################################################
        
class SpeedControlledDrone2D( system.ContinuousDynamicSystem ):
    
    def __init__(self):
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__( self, 2, 2, 2)
        
        # Labels
        self.name = 'Speed controlled planar drone'
        self.state_label = ['x','y']
        self.input_label = ['vx', 'vy']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]']
        self.input_units = ['[m/sec]', '[m/sec]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+5,+2])
        self.x_lb = np.array([-5,-2])
        
        # Kinematic param
        self.width  = 1
        self.height = 0.2
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        # Point Obstacles
        self.obstacles = np.array([[ 0.  , 20. ] ,
                                   [ 0.  , 25. ] ,
                                   [ 0.  , 30. ] ,
                                   [ 5.  , 20. ] ,
                                   [ 5.  , 25. ] ,
                                   [ 5.  , 30. ] ,
                                   [ 10. , 20. ] ,
                                   [ 10. , 25. ] ,
                                   [ 10. , 30. ] ,
                                   [ 2.5 , 35  ] ,
                                   [ 7.5 , 35  ] ] )
        
        self.is_vectorized = True
        
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
        
        dx = u       
        
        return dx
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.width * 10
        
        x = q[0]
        y = q[1]
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
                
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
        lines_style = []
        lines_color = []
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--')
        lines_color.append( 'k' )
        
        ###########################
        # drone body
        ###########################
        
        x = q[0]
        y = q[1]
        s = 0
        c = 1
        l = self.width
        h = self.height
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([x+l*c-h*s,y+l*s+h*c,0])
        pts[1,:] = np.array([x+l*c,y+l*s,0])
        pts[2,:] = np.array([x-l*c,y-l*s,0])
        pts[3,:] = np.array([x-l*c-h*s,y-l*s+h*c,0])
        
        
        lines_pts.append( pts )
        lines_style.append( 'o-')
        lines_color.append( 'b' )
        
        ###########################
        # drone cg
        ###########################
        
        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([x,y,0])
        
        lines_pts.append( pts )
        lines_style.append( 'o')
        lines_color.append( 'b' )
        
        ###########################
        # obs
        ###########################
        
        for obs in self.obstacles:

            pts = np.zeros((1,3))

            pts[0,0] = obs[0]
            pts[0,1] = obs[1]

            lines_pts.append( pts )
            lines_style.append('o')
            lines_color.append('k')
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show trust vectors
        
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # drone velocity vectors
        ###########################
        
        xcg = x[0]
        ycg = x[1]

        dx =  u[0]
        dy =  u[1]
        
        v  = np.sqrt( dx ** 2 + dy ** 2)
        
        theta = np.arctan2( dy , dx ) - np.pi * 0.5
        
        s = np.sin( theta )
        c = np.cos( theta )
        l = v * self.width * ( self.u_ub[0] - self.u_lb[0] ) 
        h = l * 0.25
        
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([xcg,ycg,0])
        pts[1,:] = np.array([xcg+dx,ycg+dy,0])
        pts[2,:] = pts[1,:] + np.array([-h*c+h*s,-h*s-h*c,0])
        pts[3,:] = pts[1,:] 
        pts[4,:] = pts[1,:] + np.array([h*c+h*s,h*s-h*c,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r' )
                
        return lines_pts , lines_style , lines_color
  


##############################################################################
# 2D planar drone
##############################################################################
    
    
##############################################################################
class ConstantSpeedHelicopterTunnel( system.ContinuousDynamicSystem ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 3   
        self.m = 1   
        self.p = 3
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Helicopter in a tunnel'
        self.state_label = ['dy','y','x']
        self.input_label = ['f']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m/sec]','[m]','[m]']
        self.input_units = ['[N]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+10,+10,+20])
        self.x_lb = np.array([-10,  0,+0])
        
        # Model param
        self.mass  = 1
        self.vx    = 10
        self.width = 1.0
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 12
        
        # Obstacles
        self.obstacles = [
                [ (2,2),(4,4)],
                [ (8,5),(10,8)],
                [ (14,0),(16,4)]
                ]

    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """

        ans = False

        for i in range(self.n):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )

        for obs in self.obstacles:
            on_obs = (( ( x[2] + self.width ) > obs[0][0]) and
                      ( ( x[1] + self.width ) > obs[0][1]) and
                      ( ( x[2] - self.width ) < obs[1][0]) and
                      ( ( x[1] - self.width ) < obs[1][1]) )

            ans = ans or on_obs

        return not(ans)
        

    #############################
    def f(self, x = np.zeros(3) , u = np.zeros(1) , t = 0 ):
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
        
        dx = np.zeros(self.n) # State derivative vector
        
        dx[0] = 1./self.mass * u[0]
        dx[1] = x[0]
        dx[2] = self.vx
        
        return dx
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   = np.array([x[2],x[1]])
        
        return q
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
        x = q[0]
        y = q[1]
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
            
                
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
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        
        ###########################
        # Vehicule
        ###########################
        
        pts      = np.zeros(( 5 , 3 ))
        
        l = self.width
        
        xyz = np.array([ q[0] , q[1] , 0 ])
        
        pts[0,:] =  np.array([-l,+l,0]) + xyz
        pts[1,:] =  np.array([+l,+l,0]) + xyz
        pts[2,:] =  np.array([+l,-l,0]) + xyz
        pts[3,:] =  np.array([-l,-l,0]) + xyz
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[1,0] = 10000
        pts[0,1] = self.x_lb[1] -l
        pts[1,1] = self.x_lb[1] -l
        
        lines_pts.append( pts )
        lines_style.append('--')
        lines_color.append('k')
        
        ###########################
        # Top line
        ###########################
            
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[1,0] = 10000
        pts[0,1] = self.x_ub[1] + l
        pts[1,1] = self.x_ub[1] + l
        
        lines_pts.append( pts )
        lines_style.append('-')
        lines_color.append('k')


        ###########################
        # obstacles
        ###########################

        for obs in self.obstacles:

            pts = np.zeros((5,3))

            pts[0,0] = obs[0][0]
            pts[0,1] = obs[0][1]

            pts[1,0] = obs[0][0]
            pts[1,1] = obs[1][1]

            pts[2,0] = obs[1][0]
            pts[2,1] = obs[1][1]

            pts[3,0] = obs[1][0]
            pts[3,1] = obs[0][1]

            pts[4,0] = obs[0][0]
            pts[4,1] = obs[0][1]

            lines_pts.append( pts )
            lines_style.append('-')
            lines_color.append('k')
            
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # force arrow
        pts      = np.zeros(( 5 , 3 ))
        
        f  = u[0] # force amplitude
        xf = x[2] # base of force x coordinate
        yf = x[1] # base of force y coordinate
        
        d = self.width / 5.0
        
        pts[0,:] =  np.array([ xf       , yf           , 0 ])
        pts[1,:] =  np.array([ xf       , yf + f       , 0 ])
        pts[2,:] =  np.array([ xf + d*f , yf + f - d*f , 0 ])
        pts[3,:] =  np.array([ xf       , yf + f       , 0 ])
        pts[4,:] =  np.array([ xf - d*f , yf + f - d*f , 0 ])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
                
        return lines_pts , lines_style , lines_color
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    #sys = Drone2D()
    
    if True:
    
        sys = Drone2D()
        
        sys.x0[5] = 0
        
        sys.ubar[0] = 9.81 * 1.1
        sys.ubar[1] = 9.81 * 0.9
        
        sys.plot_trajectory()
        sys.animate_simulation()
        
    if False:
        
        sys = SpeedControlledDrone2D()
        
        sys.x0[0] = 0
        sys.x0[1] = 0
        
        sys.ubar[0] = +0.1
        sys.ubar[1] = +3.0
        
        sys.plot_trajectory()
        sys.animate_simulation()