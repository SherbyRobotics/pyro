# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:47:05 2019

@author: alxgr
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
###############################################################################


###############################################################################
        
class Manipulator( mechanical.MechanicalSystem ):
    """ 
    Manipulator Robot 
    -------------------------------------------------------
    
    Dynamics:
    H(q) ddq + C(q,dq) dq + d(q,dq) + g(q) = B(q) u + J(q)^T f_ext
    
    Foward kinematic end-effector:
    r = foward_kinematic_effector(q)
    
    Foward differential kinematic effector:
    dr = J(q) dq
    
    Dimensions
    -------------------------------------------------------
    dof : number of degrees-of-freedom of the system
    n   : number of dynamics states (2 * dof)
    e   : number of effector dof
    m   : number of actuators inputs     
    -------------------------------------------------------
    
    Vectors
    -------------------------------------------------------
    q      :  dim = (dof, 1)   : position variables 
    dq     :  dim = (dof, 1)   : velocity variables     
    ddq    :  dim = (dof, 1)   : acceleration variables
    r      :  dim = (e, 1)     : end-effector positions
    dr     :  dim = (e, 1)     : end-effector velocities
    u      :  dim = (m, 1)     : force input variables
    f_ext  :  dim = (e, 1)     : end-effector external forces
    d      :  dim = (dof, 1)   : state-dependent dissipative forces
    g      :  dim = (dof, 1)   : state-dependent conservatives forces
    -------------------------------------------------------
    
    Matrix
    -------------------------------------------------------
    H(q)   :  dim = (dof, dof) : inertia matrix
    C(q)   :  dim = (dof, dof) : corriolis matrix
    B(q)   :  dim = (dof, m)   : actuator matrix
    J(q)   :  dim = (e , dof)  : end-effector Jacobian matrix
    -------------------------------------------------------
    
    """
    
    ############################
    def __init__(self, dof = 1 , m = 1 , e = 1):
        """ """
        
        # Effector space dimensions
        self.e = e
               
        # initialize standard params
        super().__init__( dof )
        
        # Name
        self.name = str(dof) + 'Joint Manipulator Robot'
        
        # Default Label and units
        self.effector_label = []
        self.effector_units = []
        for i in range(e):
            self.effector_label.append('Axis '+str(i))
            self.effector_units.append('[m]')
        
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    # In Mother "Mechanical System" Class
    ###########################################################################
    # def H(self, q ):
    # def C(self, q , dq ):
    # def B(self, q ):
    # def g(self, q ):
    # def d(self, q , dq ):
    ###########################################################################
    
    # Specific to "Manipulator" Class
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ 
        Foward kinematic of the end-effector
        -------------------------------------
        r : effector position vector
        q : joint space vector
        """
        
        r = np.zeros( self.e ) # Place holder
        
        return r
    
    ##############################
    def J(self, q ):
        """
        Jacobian from joint space to effector space
        -------------------------------------------
        J : e x dof
        e : nb of effector position variables
        dof: number of configuration variables
        """
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        return J
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        """ 
        External force dependent on state or time
        
        """
        
        f_ext = np.zeros( self.e ) # Default is zero vector
        
        return f_ext
    
    ###########################################################################
    # No need to overwrite the following functions for custom system
    ###########################################################################
    
    ##############################
    def forward_differential_kinematic_effector(self, q, dq ):
        """ 
        End-effector foward differential kinematic
        ------------------------------------------
        dr : time derivative of effector position vector
        """
        
        dr = np.dot( self.J(q) , dq )
        
        return dr
        
    
    ##############################
    def generalized_forces(self, q  , dq  , ddq , t = 0 ):
        """ Computed generalized forces given a trajectory """  
        
        H = self.H( q )
        C = self.C( q , dq )
        g = self.g( q )
        d = self.d( q , dq )
        
        f_ext = self.f_ext( q , dq , t )
        J     = self.J( q )
                
        # Generalized forces
        forces = ( np.dot( H , ddq ) + np.dot( C , dq ) + g + d 
                   - np.dot( J.T , f_ext ) )
        
        return forces
    
    
    ##############################
    def actuator_forces(self, q  , dq  , ddq , t = 0 ):
        """ 
        Computed actuator forces given a trajectory (inverse dynamic) 
        -------------------------------------------------------------
        Note: will fail if B is not square (under-actuated system)
        """  
        
        # Actuator Matrix
        B = self.B( q )
                
        # Generalized forces
        forces = self.generalized_forces( q , dq , ddq , t )
        
        # Actuator forces
        u = np.dot( np.linalg.inv( B ) , forces )
        
        return u
    
    
    ##############################
    def ddq(self, q , dq , u , t = 0 ):
        """ 
        Computed accelerations given actuator forces (foward dynamic)
        --------------------------------------------------------------
        
        """  
        
        H = self.H( q )
        C = self.C( q , dq )
        g = self.g( q )
        d = self.d( q , dq)
        B = self.B( q )
        
        f_ext = self.f_ext( q , dq , t )
        J     = self.J( q )
        
        ddq = np.dot( np.linalg.inv( H ) ,  ( + np.dot( B   , u )
                                              + np.dot( J.T , f_ext )
                                              - np.dot( C   , dq ) 
                                              - g 
                                              - d ) )
        return ddq
    
    
    ##############################
    def plot_end_effector_trajectory(self, traj = None ):
        """ 
        Plot the end effector trajectory with respect to time
        --------------------------------------------------------------
        
        """  
        
        # If no argument is passed, use object traj
        if traj == None:
            traj = self.traj
        
        # Plot param
        fontsize = 5
        figsize  = (4, 3)
        dpi      = 300
        
        #Number of plot
        l = self.e * 1
        
        simfig , plots = plt.subplots( l, sharex=True, figsize=figsize,
                                      dpi=dpi, frameon=True)
            
        simfig.canvas.manager.set_window_title('End-Effector trajectory for ' + 
                                       self.name)
        
        
        # Computing end-effector trajectory
        n  = traj.time_steps
        
        r_traj  = np.zeros((n,self.e))
        dr_traj = np.zeros((n,self.e))
        
        for i in range(traj.time_steps):
            
            x = traj.x[i, :]
            
            q, dq = self.x2q(x)
            
            r  = self.forward_kinematic_effector( q )
            dr = self.forward_differential_kinematic_effector(q, dq)
            
            r_traj[i,:]  =  r
            dr_traj[i,:] = dr
        
        #j = 0 # plot index
        for i in range( l ):
                plots[i].plot( traj.t , r_traj[:,i] , 'b')
                plots[i].set_ylabel(self.effector_label[i] +'\n'+
                self.effector_units[i] , fontsize = fontsize )
                plots[i].grid(True)
                plots[i].tick_params( labelsize = fontsize )
        
        plots[l-1].set_xlabel('Time [sec]', fontsize=fontsize )
        
        simfig.tight_layout()
        simfig.canvas.draw()
        plt.show()
                
        return 
        





###############################################################################
        
    
###############################################################################
        
class SpeedControlledManipulator( system.ContinuousDynamicSystem ):
    """ 
    Speed Controlled Manipulator Robot 
    -------------------------------------------------------
    
    This class can be used to model the high-level kinematic behavior of
    manipulator robot with good low-level velocity-controlled joint. All
    dynamic is neglected and inputs are considered as joint velocities.
    
    
    Dynamics:
    dq = u
    
    Foward kinematic end-effector:
    r = foward_kinematic_effector(q)
    
    Foward differential kinematic effector:
    dr = J(q) dq
    
    Dimensions
    -------------------------------------------------------
    dof : number of degrees-of-freedom of the system
    n   : number of dynamics states (2n = dof)
    e   : number of effector dof
    m   : number of actuators inputs     
    -------------------------------------------------------
    
    Vectors
    -------------------------------------------------------
    q = x  :  dim = (dof, 1)   : position variables 
    dq = u :  dim = (dof, 1)   : velocity variables     
    r      :  dim = (e, 1)     : end-effector positions
    dr     :  dim = (e, 1)     : end-effector velocities
    -------------------------------------------------------
    
    Matrix
    -------------------------------------------------------
    J(q)   :  dim = (e , dof)  : end-effector Jacobian matrix
    -------------------------------------------------------
    
    """
    ############################
    def __init__(self, dof , e ):
        """ """
        
        # Effector space dimensions
        self.e = e

        # Degree of Freedom
        self.dof = dof
        
        # Nb of states
        n = dof
        
        # initialize standard params
        super().__init__(dof, dof, dof)
        
        # Name
        self.name = str(n) + ' Joint Speed Controlled Manipulator'
        
        # Labels, bounds and units
        for i in range(n):
            # joint angle states
            self.x_ub[i] = + np.pi * 2
            self.x_lb[i] = - np.pi * 2
            self.state_label[i] = 'Angle '+ str(i)
            self.state_units[i] = '[rad]'
        for i in range(n):
            # joint velocity inputs
            self.u_ub[i] = + np.pi * 2
            self.u_lb[i] = - np.pi * 2
            self.input_label[i] = 'Velocity ' + str(i)
            self.input_units[i] = '[rad/sec]'
            
    
    ############################
    @classmethod
    def from_manipulator(cls, Manipulator):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( Manipulator.dof , Manipulator.e )
        
        instance.forward_kinematic_lines    = Manipulator.forward_kinematic_lines
        instance.forward_kinematic_domain   = Manipulator.forward_kinematic_domain
        instance.forward_kinematic_effector = Manipulator.forward_kinematic_effector
        instance.J                          = Manipulator.J
        instance.isavalidstate              = Manipulator.isavalidstate
        
        return instance
        
            
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
    
    ##############################
    def forward_differential_kinematic_effector(self, q, dq ):
        """ 
        End-effector foward differential kinematic
        ------------------------------------------
        dr : time derivative of effector position vector
        """
        
        dr = np.dot( self.J(q) , dq )
        
        return dr
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ 
        Foward kinematic of the end-effector
        -------------------------------------
        r : effector position vector
        q : joint space vector
        """
        
        r = np.zeros( self.e ) # Place holder
        
        return r
    
    ##############################
    def J(self, q ):
        """
        Jacobian from joint space to effector space
        -------------------------------------------
        J : e x dof
        e : nb of effector position variables
        dof: number of configuration variables
        """
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        return J
            


###############################################################################
# One Link Manipulator
###############################################################################
        
class OneLinkManipulator( Manipulator ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
        
        # Dimensions
        dof = 1
        m   = 1
        e   = 2
               
        # initialize standard params
        super().__init__(dof , m , e)
        
        # Name
        self.name = 'One Link Manipulator'
        
        # Plot param
        self.linestyle = '-'
                
        # Model parameters
        self.l1      = 2.5  # Link length
        self.lc1     = 1.2  # Center of mass distance from pivot
        self.m1      = 1    # Mass
        self.I1      = 0    # Inertia
        self.gravity = 9.81 # Gravity field constant
        self.d1      = 0.1  # Linear damping coef
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        
        return [c1,s1]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        
        [c1,s1] = self.trig( q )
        
        x = self.l1 * s1 
        y = self.l1 * c1 
        
        r = np.array([x,y])
        
        return r
    
    
    ##############################
    def J(self, q ):
        
        [c1,s1] = self.trig( q )
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        J[0] = + self.l1 * c1 
        J[1] = - self.l1 * s1 
        
        return J
    
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        
        f_ext = np.zeros( self.e ) # Default is zero vector
        
        return f_ext
        
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        [c1,s1] = self.trig( q )
        
        H = np.zeros((1,1))
        
        H[0,0] = ( self.m1 * self.lc1**2 + self.I1  )
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((1,1))
        
        C[0,0] = 0

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
        
        [c1,s1] = self.trig( q )
        
        g1 = (self.m1 * self.lc1  ) * self.gravity
        
        g = np.zeros(1)
        
        g[0] = - g1 * s1 

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((1,1))
        
        D[0,0] = self.d1
        
        d = np.dot( D , dq )
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.l1 * 1.2
        
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
        
        pts      = np.zeros(( 6 , 3 ))
        
        [c1,s1] = self.trig( q )
        
        l = self.l1 * 0.9
        h = self.l1 * 0.1
        
        pts[1,0] = 0 * s1 + h * c1
        pts[1,1] = 0 * c1 - h * s1
        
        pts[2,0] = l * s1 + h * c1
        pts[2,1] = l * c1 - h * s1
        
        pts[3,0] = l * s1 - h * c1
        pts[3,1] = l * c1 + h * s1
        
        pts[4,0] = 0 * s1 - h * c1
        pts[4,1] = 0 * c1 + h * s1
        
        lines_pts.append( pts )
        
        
        ###########################
        # end effector
        ###########################
        
        pts      = np.zeros(( 7 , 3 ))
        
        pts[0,0] = l * s1 + 0 * c1
        pts[0,1] = l * c1 - 0 * s1
        
        pts[1,0] = (l+h) * s1 + 0 * c1
        pts[1,1] = (l+h) * c1 - 0 * s1
        
        pts[2,0] = (l+h) * s1 - h * c1
        pts[2,1] = (l+h) * c1 + h * s1
        
        pts[3,0] = (l+h+h) * s1 - h * c1
        pts[3,1] = (l+h+h) * c1 + h * s1
        
        pts[4,0] = (l+h) * s1 - h * c1
        pts[4,1] = (l+h) * c1 + h * s1
        
        pts[5,0] = (l+h) * s1 + h * c1
        pts[5,1] = (l+h) * c1 - h * s1
        
        pts[6,0] = (l+h+h) * s1 + h * c1
        pts[6,1] = (l+h+h) * c1 - h * s1
        
        
        lines_pts.append( pts )
                
        return lines_pts





###############################################################################
# Two Link Manipulator
###############################################################################
        
class TwoLinkManipulator( Manipulator ):
    """  """
    
    ############################
    def __init__(self):
        """ """
        
        # Dimensions
        dof = 2
        m   = 2
        e   = 2
               
        # initialize standard params
        super().__init__(dof , m , e)
        
        # Name
        self.name = 'Two Link Manipulator'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 0.5
        self.l2  = 0.3
        self.lc1 = 0.2
        self.lc2 = 0.1
        
        self.m1 = 1
        self.I1 = 0
        self.m2 = 1
        self.I2 = 0
        
        self.gravity = 9.81
        
        self.d1 = 0.5
        self.d2 = 0.5
        
        
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
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        x = self.l1 * s1 + self.l2 * s12 # x
        y = self.l1 * c1 + self.l2 * c12 # y
        
        r = np.array([x,y])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        J[0,0] = + self.l1 * c1 + self.l2 * c12
        J[0,1] =                  self.l2 * c12
        J[1,0] = - self.l1 * s1 - self.l2 * s12
        J[1,1] =                - self.l2 * s12
        
        return J
    
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        """ """
        
        f_ext = np.zeros( self.e ) # Default is zero vector
        
        return f_ext
        
    
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
        
        H[0,0] = ( self.m1 * self.lc1**2 + self.I1 + self.m2 * ( self.l1**2 
                 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2 ) + self.I2 )
        H[1,0] = ( self.m2 * self.lc2**2 + self.m2 * self.l1 * self.lc2 * c2 
                   + self.I2 )
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
        l = 1
        
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
# Three Link Manipulator
###############################################################################
        
class ThreeLinkManipulator3D( Manipulator ):
    """ 
    Three link Manipulator Class 
    -------------------------------
    
    base:     revolute arround z
    shoulder: revolute arround y
    elbow:    revolute arround y
    
    see Example 4.3 in
    http://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-manipdyn_v1_2.pdf
    """
    
    ############################
    def __init__(self):
        """ """
        
        # Dimensions
        dof = 3
        m   = 3
        e   = 3
               
        # initialize standard params
        super().__init__(dof , m , e)
        
        # Name
        self.name = 'Three Link Manipulator'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        # Kinematic
        self.l1  = 1 
        self.l2  = 1
        self.l3  = 1
        self.lc1 = 1
        self.lc2 = 1
        self.lc3 = 1
        
        # Inertia
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1
        
        self.I1z = 1
        
        self.I2x = 1
        self.I2y = 1
        self.I2z = 1
        
        self.I3x = 1
        self.I3y = 1
        self.I3z = 1
        
        # Gravity
        self.gravity = 9.81
        
        # Joint damping
        self.d1 = 1
        self.d2 = 1
        self.d3 = 1
        
        # Total length
        self.lw  = (self.l1+self.l2+self.l3)
        
        
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
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # Three robot points
        
        # Base of the robot
        p0 = [0,0,0]
        
        # Shperical point 
        p1 = [ 0, 0, self.l1 ]
        
        # Elbow
        z2 = self.l1 - self.l2 * s2
        
        r2 = self.l2 * c2
        x2 = r2 * c1
        y2 = r2 * s1
        
        p2 = [ x2, y2, z2 ]
        
        # End-effector
        z3 = self.l1 - self.l2 * s2 - self.l3 * s23
        
        r3 = self.l2 * c2 + self.l3 * c23
        x3 = r3 * c1
        y3 = r3 * s1
                
        r = np.array([x3, y3, z3])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        J = np.zeros((3,3))
        
        J[0,0] =  -( self.l2 * c2 + self.l3 * c23 ) * s1
        J[0,1] =  -( self.l2 * s2 + self.l3 * s23 ) * c1
        J[0,2] =  - self.l3 * s23 * c1
        J[1,0] =   ( self.l2 * c2 + self.l3 * c23 ) * c1
        J[1,1] =  -( self.l2 * s2 + self.l3 * s23 ) * s1
        J[1,2] =  - self.l3 * s23 * s1
        J[2,0] =  0
        J[2,1] =  -( self.l2 * c2 + self.l3 * c23 )
        J[2,2] =  - self.l3 * c23
        
        return J
    
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        """ """
        
        f_ext = np.zeros( self.e ) # Default is zero vector
        
        return f_ext
        
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # variable to match the book notation
        
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        
        Iz1 = self.I1z
        Ix2 = self.I2x
        Iy2 = self.I2y
        Iz2 = self.I2z
        Ix3 = self.I3x
        Iy3 = self.I3y
        Iz3 = self.I3z
        
        l1 = self.l2
        r1 = self.lc2
        l2 = self.l3
        r2 = self.lc3
        
        
        H = np.zeros((3,3))
        
        H[0,0] = (Iy2 * s2 **2 + Iy3 * s23 **2 + Iz1 
                 + Iz2 * c2 **2 + Iz3 * c23 **2 + m2 * ( r1 * c2 ) **2 
                 + m3 * ( l1 * c2 + r2 * c23 ) **2 )
        H[0,1] = 0
        H[0,2] = 0
        H[1,0] = 0
        H[1,1] = (Ix2 + Ix3 + m3 * l1 **2 + m2 * r1 **2 
                 + m3 * r2 **2 + 2 * m3 *l1 * r2 * c3)
        H[1,2] = Ix3 + m3 * r2 **2 + m3 * l1 * r2 * c3
        H[2,0] = 0
        H[2,1] = H[1,2]
        H[2,2] = Ix3 + m3 * r2 ** 2
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # variable to match the book notation
        
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        
        Iz1 = self.I1z
        Ix2 = self.I2x
        Iy2 = self.I2y
        Iz2 = self.I2z
        Ix3 = self.I3x
        Iy3 = self.I3y
        Iz3 = self.I3z
        
        l1 = self.l2
        r1 = self.lc2
        l2 = self.l3
        r2 = self.lc3
        
        
        T112 = (( Iy2 - Iz2 - m2 * r1 **2 ) * c2 * s2 
                 + ( Iy3 - Iz3 ) * c23 * s23
                 - m3 * ( l1 * c2 + r2 * c23 ) * ( l1 * s2 + r2 * s23 ) )
        T113 = (( Iy3 - Iz3 ) * c23 * s23 
               - m3 * r2 * s23 * ( l1 * c2 + r2 * c23 ))
        T121 = T112
        T131 = T113
        T211 = (( Iz2 - Iy2 + m2 * r1 **2 ) * c2 * s2 
               + ( Iz3 - Iy3 ) * c23 * s23 + m3 * 
                ( l1 * c2 + r2 * c23 ) * ( l1 * s2 + r2 * s23 ))
        T223 = - l1 * m3 * r2 * s3
        T232 = T223
        T233 = T223
        T311 = (( Iz3 - Iy3 ) * c23 * s23 
                + m3 * r2 * s23 * ( l1 * c2 + r2 * c23 ))
        T322 = l1 * m3 * r2 * s3
                
        C = np.zeros((3,3))
        
        C[0,0] = T112 * dq[1] + T113 * dq[2]
        C[0,1] = T121 * dq[0]
        C[0,2] = T131 * dq[0]
        
        C[1,0] = T211 * dq[0]
        C[1,1] = T223 * dq[2]
        C[1,2] = T232 * dq[1] + T233 * dq[2]
        
        C[2,0] = T311 * dq[0]
        C[2,1] = T322 * dq[1]
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
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        G = np.zeros(3)
        
        g = self.gravity
        
        G[0] = 0
        G[1] = -(self.m2 * g * self.lc2 + self.m3 * g * self.l2 ) * c2 - self.m3 * g * self.lc3 * c23
        G[2] = - self.m3 * g * self.lc3 * c23

        return G
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((3,3))
        
        D[0,0] = self.d1
        D[1,1] = self.d2
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
        
        domain  = [ (-l,l) , (-l,l) , (0,l*2) ]#  
                
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
        
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([-1,-1,0])
        pts[1,:] = np.array([+1,-1,0])
        pts[2,:] = np.array([+1,+1,0])
        pts[3,:] = np.array([-1,+1,0])
        pts[4,:] = np.array([-1,-1,0])
        
        lines_pts.append( pts )
        
        ###########################
        # robot kinematic
        ###########################
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # Three robot points

        # Shperical point 
        pts[1,0] = 0
        pts[1,1] = 0
        pts[1,2] = self.l1 
        
        # Elbow
        z2 = self.l1 - self.l2 * s2
        
        r2 = self.l2 * c2
        x2 = r2 * c1
        y2 = r2 * s1
        
        pts[2,0] = x2
        pts[2,1] = y2
        pts[2,2] = z2
        
        # End-effector
        z3 = self.l1 - self.l2 * s2 - self.l3 * s23
        
        r3 = self.l2 * c2 + self.l3 * c23
        x3 = r3 * c1
        y3 = r3 * s1
                
        pts[3,0] = x3
        pts[3,1] = y3
        pts[3,2] = z3 
        
        lines_pts.append( pts )
                
        return lines_pts



###############################################################################
# Five Planar Link Manipulator
###############################################################################
        
class FiveLinkPlanarManipulator( Manipulator ):
    """ 
    Note: Only kinematic functions 
    """
    
    ############################
    def __init__(self):
        """ """
        
        # Dimensions
        dof = 5
        m   = 5
        e   = 2
               
        # initialize standard params
        super().__init__(dof , m , e)
        
        # Name
        self.name = 'Five Link Manipulator'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        l1  = 0.5
        l2  = 0.5
        l3  = 0.5
        l4  = 0.5
        l5  = 0.5
        
        self.l = np.array([l1,l2,l3,l4,l5])
        
        
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
        c4  = np.cos( q[3] )
        s4  = np.sin( q[3] )
        c5  = np.cos( q[4] )
        s5  = np.sin( q[4] )
        
        cos_rel = np.array( [ c1 , c2 , c3 , c4 , c5 ])
        sin_rel = np.array( [ s1 , s2 , s3 , s4 , s5 ])
        
        c12    = np.cos( q[0] + q[1] )
        s12    = np.sin( q[0] + q[1] )
        c123   = np.cos( q[0] + q[1] + q[2])
        s123   = np.sin( q[0] + q[1] + q[2])
        c1234  = np.cos( q[0] + q[1] + q[2] + q[3])
        s1234  = np.sin( q[0] + q[1] + q[2] + q[3])
        c12345 = np.cos( q[0] + q[1] + q[2] + q[3] + q[4])
        s12345 = np.sin( q[0] + q[1] + q[2] + q[3] + q[4])
        
        cos_abs = np.array( [ c1 , c12 , c123 , c1234 , c12345 ])
        sin_abs = np.array( [ s1 , s12 , s123 , s1234 , s12345 ])
        
        return [cos_rel,sin_rel,cos_abs,sin_abs]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        x = (self.l * sin_abs).sum()
        y = (self.l * cos_abs).sum()
        
        r = np.array([x,y])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        J[0,0] = self.l[4] * cos_abs[4] + self.l[3] * cos_abs[3] + self.l[2] * cos_abs[2] + self.l[1] * cos_abs[1]  + self.l[0] * cos_abs[0] 
        J[0,1] = self.l[4] * cos_abs[4] + self.l[3] * cos_abs[3] + self.l[2] * cos_abs[2] + self.l[1] * cos_abs[1]             
        J[0,2] = self.l[4] * cos_abs[4] + self.l[3] * cos_abs[3] + self.l[2] * cos_abs[2]           
        J[0,3] = self.l[4] * cos_abs[4] + self.l[3] * cos_abs[3]                
        J[0,4] = self.l[4] * cos_abs[4] 
        
        J[1,0] = - self.l[4] * sin_abs[4] - self.l[3] * sin_abs[3] - self.l[2] * sin_abs[2] - self.l[1] * sin_abs[1]  - self.l[0] * sin_abs[0] 
        J[1,1] = - self.l[4] * sin_abs[4] - self.l[3] * sin_abs[3] - self.l[2] * sin_abs[2] - self.l[1] * sin_abs[1]             
        J[1,2] = - self.l[4] * sin_abs[4] - self.l[3] * sin_abs[3] - self.l[2] * sin_abs[2]           
        J[1,3] = - self.l[4] * sin_abs[4] - self.l[3] * sin_abs[3]                
        J[1,4] = - self.l[4] * sin_abs[4] 
        
        return J
    
        
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
        # robot kinematic
        ###########################
        
        pts      = np.zeros(( 6 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        pts[1,0] = self.l[0] * sin_abs[0]
        pts[2,0] = self.l[1] * sin_abs[1] + pts[1,0]
        pts[3,0] = self.l[2] * sin_abs[2] + pts[2,0]
        pts[4,0] = self.l[3] * sin_abs[3] + pts[3,0]
        pts[5,0] = self.l[4] * sin_abs[4] + pts[4,0]
        
        pts[1,1] = self.l[0] * cos_abs[0]
        pts[2,1] = self.l[1] * cos_abs[1] + pts[1,1]
        pts[3,1] = self.l[2] * cos_abs[2] + pts[2,1]
        pts[4,1] = self.l[3] * cos_abs[3] + pts[3,1]
        pts[5,1] = self.l[4] * cos_abs[4] + pts[4,1]
        
        lines_pts.append( pts )
                
        return lines_pts



##############################################################################
# Two Planar Link Manipulator with obstacles
###############################################################################
        
class TwoLinkManipulatorwithObstacles( TwoLinkManipulator ):
    """
    Maniplator with non-allowable states based on end-effector position only
    """
    
    ############################
    def __init__(self):
        """ """
        # initialize standard params
        super().__init__()
        
        self.l1 = 1.1
        self.l2 = 0.9
        
        # Labels
        self.name = 'Two Link Planar Manipulator with Obstacles'

        self.obstacles = [
                [ (-1, -1),( 3, -0.2)],
                [ (0.5, 0.5),(2, 3)],
                [ (-2, 0),(-0.2, 3)]
                ]
        
        
    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """
        ans = False
        for i in range( x.size ):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )
        
        # effector position
        q = self.xut2q( x, self.ubar, 0)
        r = self.forward_kinematic_effector( q )

        buffer = 0.0
        
        for obs in self.obstacles:
            on_obs = (( r[0] + buffer > obs[0][0]) and  
                      ( r[1] + buffer > obs[0][1]) and 
                      ( r[0] - buffer < obs[1][0]) and 
                      ( r[1] - buffer < obs[1][1]) )
                     
            ans = ans or on_obs
            
        return not(ans)
    
        
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
        
        lines_pts = TwoLinkManipulator.forward_kinematic_lines(self, q )
        

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

    ###########################################################################


###############################################################################
# Five Planar Link Manipulator with obstacles
###############################################################################
        
class FiveLinkPlanarManipulatorwithObstacles( FiveLinkPlanarManipulator ):
    """
    Maniplator with non-allowable states based on end-effector position only
    """
    
    ############################
    def __init__(self):
        """ """
        # initialize standard params
        super().__init__()
        
        # Labels
        self.name = 'Five Link Planar Manipulator with Obstacles'

        self.obstacles = [
                [ (-1, -1),( 3, -0.2)],
                [ (0.5, 0.5),(2, 3)],
                [ (-2, 0),(-0.2, 3)]
                ]
        
        
    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """
        ans = False
        for i in range( x.size ):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )
        
        # effector position
        q = self.xut2q( x, self.ubar, 0)
        r = self.forward_kinematic_effector( q )

        buffer = 0.0
        
        for obs in self.obstacles:
            on_obs = (( r[0] + buffer > obs[0][0]) and  
                      ( r[1] + buffer > obs[0][1]) and 
                      ( r[0] - buffer < obs[1][0]) and 
                      ( r[1] - buffer < obs[1][1]) )
                     
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
        
        lines_pts = FiveLinkPlanarManipulator.forward_kinematic_lines(self, q )
        

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

    ###########################################################################
    
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    
    sys = OneLinkManipulator()
    #sys = TwoLinkManipulator()
    
    sys.x0[0] = 0.1
    sys.animate_simulation()
    sys.plot_trajectory()
    sys.plot_end_effector_trajectory()
    
    
    #sys = ThreeLinkManipulator3D()
    #sys.x0[0] = 0.1
    #sys.animate_simulation( is_3d = True )
    #sys.plot_trajectory()
    
    #sys = FiveLinkPlanarManipulator()
    #sys.ubar = np.array([1,1,1,1,1])
    #sys.animate_simulation()
    
    #aaa = TwoLinkManipulatorwithObstacles()
    #bbb = SpeedControlledManipulator.from_manipulator( aaa )
    #bbb.ubar = np.array([1,1])
    #bbb.animate_simulation()