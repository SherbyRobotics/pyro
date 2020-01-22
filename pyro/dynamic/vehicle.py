# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:45:37 2018

@author: Alexandre
"""


###############################################################################
import numpy as np
##############################################################################
from pyro.dynamic import system
###############################################################################


##############################################################################
#
##############################################################################
        
class KinematicBicyleModel( system.ContinuousDynamicSystem ):
    """ 
    Equations of Motion
    -------------------------
    dx   = V cos ( phi )
    dy   = V sin ( phi )
    dphi = V/l tan ( beta )
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 3   
        self.m = 2   
        self.p = 3
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Kinematic Bicyle Model'
        self.state_label = ['x','y','phi']
        self.input_label = ['v', 'beta']
        self.output_label = ['x','y','phi']
        
        # Units
        self.state_units = ['[m]','[m]','[rad]']
        self.input_units = ['[m/sec]', '[rad]']
        self.output_units = ['[m]','[m]','[rad]']
        
        # State working range
        self.x_ub = np.array([+5,+2,+3.14])
        self.x_lb = np.array([-5,-2,-3.14])
        
        # Model param
        self.lenght = 1
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
    #############################
    def f(self, x = np.zeros(3) , u = np.zeros(2) , t = 0 ):
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

        dx[0] = u[0] * np.cos( x[2] )
        dx[1] = u[0] * np.sin( x[2] )
        dx[2] = u[0] * np.tan( u[1] ) * ( 1. / self.lenght) 
        
        return dx
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   = np.append(  x , u[1] ) # steering angle is part of the config
        
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
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        ###########################
        # Top line
        ###########################
            
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[0,1] = 1
        pts[1,0] = 10000
        pts[1,1] = 1
        
        lines_pts.append( pts )
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[0,1] = -1
        pts[1,0] = 10000
        pts[1,1] = -1
        
        lines_pts.append( pts )
        
        ###########################
        # Vehicule
        ###########################
        
        pts = np.zeros((3,3))
        
        pts[0,0] = q[0]
        pts[0,1] = q[1]
        pts[1,0] = q[0] + self.lenght * np.cos( q[2] )
        pts[1,1] = q[1] + self.lenght * np.sin( q[2] )
        pts[2,0] = ( q[0] + self.lenght * np.cos( q[2] ) + 
                       0.2 * self.lenght * np.cos( q[2] + q[3] ) )
        pts[2,1] = ( q[1] + self.lenght * np.sin( q[2] ) + 
                       0.2 * self.lenght * np.sin( q[2] + q[3] ) )
        
        lines_pts.append( pts )
                
        return lines_pts
    
    

##############################################################################
#
##############################################################################
        
class HolonomicMobileRobot( system.ContinuousDynamicSystem ):
    """ 
    
    dx   = u[0]
    dy   = u[1]
    
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 2   
        self.m = 2   
        self.p = 2
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Holonomic Mobile Robot'
        self.state_label = ['x','y']
        self.input_label = ['vx', 'vy']
        self.output_label = ['x','y']
        
        # Units
        self.state_units = ['[m]','[m]']
        self.input_units = ['[m/sec]','[m/sec]']
        self.output_units = ['[m]','[m]']
        
        # State working range
        self.x_ub = np.array([ 10, 10])
        self.x_lb = np.array([-10,-10])
        
    #############################
    def f(self, x = np.zeros(3) , u = np.zeros(2) , t = 0 ):
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

        dx[0] = u[0]
        dx[1] = u[1] 
        
        return dx
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q = x # kinematic model : state = config space
        
        return q
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 10
        
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
        
        ###########################
        # Top line
        ###########################
            
        pts = np.zeros((4,3))
        
        d = 0.2
        
        pts[0,0] = q[0]+d
        pts[0,1] = q[1]+d
        pts[1,0] = q[0]+d
        pts[1,1] = q[1]-d
        pts[2,0] = q[0]-d
        pts[2,1] = q[1]-d
        pts[3,0] = q[0]-d
        pts[3,1] = q[1]+d
        
        lines_pts.append( pts )
                
        return lines_pts


##############################################################################
#
##############################################################################
        
class HolonomicMobileRobotwithObstacles( HolonomicMobileRobot ):
    """ 
    
    dx   = u[0]
    dy   = u[1]
    
    """
    
    ############################
    def __init__(self):
        """ """
        # initialize standard params
        HolonomicMobileRobot.__init__(self)
        
        # Labels
        self.name = 'Holonomic Mobile Robot with Obstacles'

        # State working range
        self.x_ub = np.array([ 10, 10])
        self.x_lb = np.array([-10,-10])
        
        self.obstacles = [
                [ (2,2),(4,10)],
                [ (6,-8),(8,8)],
                [ (-8,-8),(-1,8)]
                ]
        
    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """
        
        ans = False
        
        for i in range(self.n):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )
        
        for obs in self.obstacles:
            on_obs = (( x[0] > obs[0][0]) and  
                      ( x[1] > obs[0][1]) and 
                      ( x[0] < obs[1][0]) and 
                      ( x[1] < obs[1][1]) )
                     
            ans = ans or on_obs
            
        return not(ans)
        
       
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
        
        ###########################
        # Vehicule
        ###########################
            
        pts = np.zeros((4,3))
        
        d = 0.2
        
        pts[0,0] = q[0]+d
        pts[0,1] = q[1]+d
        pts[1,0] = q[0]+d
        pts[1,1] = q[1]-d
        pts[2,0] = q[0]-d
        pts[2,1] = q[1]-d
        pts[3,0] = q[0]-d
        pts[3,1] = q[1]+d
        
        lines_pts.append( pts )
        
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
            
                
        return lines_pts
    
    
##############################################################################
#
##############################################################################
        
class KinematicCarModel( KinematicBicyleModel ):
    """ 
    
    Bicycle model with car graphical output
    
    Equations of Motion
    -------------------------
    dx   = V cos ( phi )
    dy   = V sin ( phi )
    dphi = V/l tan ( beta )
    
    """
    
    ############################
    def __init__(self):
        """ """
        
        # initialize standard params
        KinematicBicyleModel.__init__(self,)
        
        # Model param
        self.width  = 2.00
        self.a      = 2.00
        self.b      = 3.00
        self.lenght = self.a+self.b    
        self.lenght_tire = 0.40
        self.width_tire = 0.15
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = self.lenght * 2
        
        
    ###########################################################################
    # For graphical output
    ###########################################################################

    
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
        
        ###########################
        # Top line
        ###########################
            
        pts = np.zeros((2,10))
        
        pts[0,0] = -10000
        pts[0,1] = self.width * 2.25
        pts[1,0] = 10000
        pts[1,1] = self.width * 2.25
        
        lines_pts.append( pts )
        
        ###########################
        # middle line
        ###########################
        
        pts = np.zeros((2,10))
        
        pts[0,0] = -10000
        pts[0,1] = self.width * 0.75
        pts[1,0] = 10000
        pts[1,1] = self.width * 0.75
        
        lines_pts.append( pts )
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,10))
        
        pts[0,0] = -10000
        pts[0,1] = - self.width * 0.75
        pts[1,0] = 10000
        pts[1,1] = - self.width * 0.75
        
        lines_pts.append( pts )
        
        ###########################
        # Car
        ###########################
        
        """
        Here is how the car is drawn:
        
        |---------lenght--------|
        |-----a------|
        *                       *   -
            *               *       |
          d2    *   CG  *    d3     |
                    *              width
          d1    *       *    d4     |
           *                *       |
        *                       *   -
        """
        # Distance of the four corners of the car from the mass center
        d1 = np.sqrt(self.a**2+(self.width/2)**2)
        d3 = np.sqrt((self.lenght-self.a)**2+(self.width/2)**2)
        
        # Angles of the four lines of the car       
        theta1 = np.pi+np.arctan(self.width/2/self.a)+q[2]
        theta2 = np.pi-np.arctan(self.width/2/self.a)+q[2]
        theta3 = np.arctan(self.width/2/(self.lenght-self.a))+q[2]
        theta4 = 2*np.pi-np.arctan(self.width/2/(self.lenght-self.a))+q[2]
        
        # Build first line of the car 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d1,d1,theta1,theta2,q[0],q[1])
        lines_pts.append( pts )
        
        # Build second line of the car 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d3,d3,theta3,theta4,q[0],q[1])
        px3 = pts[0,0] # Points used to center the wheels
        py3 = pts[0,1]
        px4 = pts[1,0]
        py4 = pts[1,1]
        lines_pts.append( pts )
        
        # Build third line of the car 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d1,d3,theta2,theta3,q[0],q[1])
        lines_pts.append( pts )

        # Build third fourth of the car 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d1,d3,theta1,theta4,q[0],q[1])
        lines_pts.append( pts )
        
        ###########################
        # Wheels
        ###########################
        
        # Distance of the four corners of a tire from the center
        d  = np.sqrt((self.lenght_tire/2)**2+(self.width_tire/2)**2)
        
        # Angles of the four lines of a tire               
        steer1 = np.pi+np.arctan(self.width/2/self.a)+q[3]+q[2]
        steer2 = np.pi-np.arctan(self.width/2/self.a)+q[3]+q[2]
        steer3 = np.arctan(self.width/2/(self.lenght-self.a))+q[3]+q[2]
        steer4 = 2*np.pi-np.arctan(self.width/2/(self.lenght-self.a))+q[3]+q[2]
        
        # Build first line of the left tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer1,steer2,px3,py3)
        lines_pts.append( pts )
        
        # Build second line of the left tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer3,steer4,px3,py3)
        lines_pts.append( pts )

        # Build third line of the left tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer2,steer3,px3,py3)
        lines_pts.append( pts )
        
        # Build fourth line of the left tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer1,steer4,px3,py3)
        lines_pts.append( pts )
        
        # Build first line of the right tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer1,steer2,px4,py4)
        lines_pts.append( pts )
        
        # Build second line of the right tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer3,steer4,px4,py4)
        lines_pts.append( pts )
        
        # Build third line of the right tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer2,steer3,px4,py4)
        lines_pts.append( pts )

        # Build first line of the right tire 
        pts = np.zeros((2,10))
        pts[0,0],pts[0,1],pts[1,0],pts[1,1] = self.draw_line(d,d,steer1,steer4,px4,py4)
        lines_pts.append( pts )
                
        return lines_pts
        
    ##########################################################################        
    def draw_line(self, d1, d2, angle1, angle2, x, y):
        
        x1 = x+d1*np.cos(angle1)
        y1 = y+d1*np.sin(angle1)
        x2 = x+d2*np.cos(angle2)
        y2 = y+d2*np.sin(angle2)
            
        return x1,y1,x2,y2
    
    
 ##############################################################################       
class KinematicCarModelwithObstacles( KinematicCarModel ):
    """ 
    
    """
    
    ############################
    def __init__(self):
        """ """
        # initialize standard params
        KinematicCarModel.__init__(self)
        
        # Labels
        self.name = 'Kinematic Bicyle Model with Obstacles'
        
        self.obstacles = [
                [ (-10,-1),(-5,1)],
                [ (-4,2),(1,4)],
                [ (12,-1),(17,1)]
                ]
        
    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """
        ans = False
        for i in range(self.n):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )
        
        for obs in self.obstacles:
            on_obs = (( x[0] + self.lenght * 0.5 > obs[0][0]) and  
                      ( x[1] + self.width * 0.5  > obs[0][1]) and 
                      ( x[0] - self.lenght * 0.5 < obs[1][0]) and 
                      ( x[1] - self.width * 0.5 < obs[1][1]) )
                     
            ans = ans or on_obs
            
        return not(ans)
        
       
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = KinematicCarModel.forward_kinematic_lines(self, q )
        

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
            
                
        return lines_pts   

        
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    #sys = KinematicBicyleModel()
    #sys = KinematicCarModel()
    sys = KinematicCarModelwithObstacles()
    
    sys.ubar = np.array([2,-0.5])
    sys.plot_trajectory( np.array([0,0,0]) , 10 )
    
    sys.animate_simulation( 1 )
        