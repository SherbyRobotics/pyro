# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:43:21 2019

@author: alxgr
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control import controller
###############################################################################


###############################################################################
# Mother Class
###############################################################################

class RobotController( controller.StaticController ) :
    """   
    Mother class for robot controllers
    ---------------------------------------
    r  : reference signal vector       k x 1
    y  : sensor signal vector          p x 1
    u  : control inputs vector         m x 1
    t  : time                          1 x 1
    ---------------------------------------
    u = c( y , r , t )
    """
    
    ############################
    def __init__(self, dof = 1):
        """ """

        # Dimensions
        self.k = dof      # ref signal dimension
        self.m = dof      # assuming fully-actuated system
        self.p = dof * 2  # y = x ( full state feedback )
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Robot Controller'
        
        # DoF
        self.dof = dof
        
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        :     self.dof   ]
        dq = x[ self.dof : 2 * self.dof   ]
        
        return [ q , dq ]
        


###############################################################################
# Joint PD
###############################################################################
        
class JointPD( RobotController ) :
    """ 
    Linear controller for mechanical system with full state feedback (y=x)
    Independent PID for each DOF
    ---------------------------------------
    r  : reference signal vector  dof x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd 

    """
    
    ############################
    def __init__(self, dof = 1, kp = 1, kd = 0):
        """ """
        
        super().__init__( dof )
        
        # Label
        self.name = 'Joint PD Controller'
        
        # Gains
        self.kp = np.ones( dof  ) * kp
        self.kd = np.ones( dof  ) * kd
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        de =     - dq
        
        # PIDs
        u = e * self.kp + de * self.kd 
        
        return u
    


###############################################################################
# End-Effector PD
###############################################################################
        
class EndEffectorPD( RobotController ) :
    """ 
    PID in effector coordinates, using the Jacobian of the system
    ---------------------------------------
    r  : reference signal vector  e   x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd 

    """
    
    ############################
    def __init__(self, robot, kp = 1, ki = 0, kd = 0):
        """ """
        
        # Using from model
        self.fwd_kin = robot.forward_kinematic_effector
        self.J       = robot.J
        self.e       = robot.e # nb of effector dof
        
        super().__init__( robot.dof )
        
        # Label
        self.name = 'End-Effector PID Controller'
        
        # Gains
        self.kp = np.ones( self.dof  ) * kp
        self.kd = np.ones( self.dof  ) * kd
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        dr          = np.dot( J , dq )
        
        # Error
        e  = r_desired - r_actual
        de =           - dr
        
        # Effector space PID
        f = e * self.kp + de * self.kd 
        
        # From effector force to joint torques
        u = np.dot( J.T , f )
        
        return u
    
    
###############################################################################
# Kinematic Controllers
###############################################################################
        
class EndEffectorKinematicController( RobotController ) :
    """ 
    Kinematic effector coordinates controller using the Jacobian of the system
    ------------------------------------------
    r = r_d : reference signal vector  e   x 1
    y = q   : sensor signal vector     dof x 1
    u = dq  : control inputs vector    dof x 1
    t       : time                     1   x 1
    -------------------------------------------
    u = c( y , r , t ) = J(q)^T *  [ (r - r_robot(q)) * k ]

    """
    
    ############################
    def __init__(self, robot, k = 1 ):
        """ """
        
        # Using functions from robot model
        self.fwd_kin = robot.forward_kinematic_effector
        self.J       = robot.J
        self.e       = robot.e # nb of effector dof
        
        # Dimensions
        self.dof = robot.dof
        self.k   = self.e 
        self.m   = self.dof
        self.p   = self.dof
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)

        # Label
        self.name = 'End Effector Kinematic Controller'
        
        # Gains
        self.gains = np.ones( self.e  ) * k
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        #u = np.zeros(self.m) 
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        # Effector space speed
        dr_r = e * self.gains
        
        # From effector speed to joint speed
        if self.dof == self.e:
            
            dq_r = np.dot( np.linalg.inv( J ) , dr_r )
            
        elif self.dof > self.e:
            
            # Pseudo-inverse of Jacobian
            J_pinv = np.linalg.pinv( J )
            
            dq_r   = np.dot( J_pinv , dr_r )
            
        else:
            
            #TODO
            pass
        
        return dq_r
    
    
class EndEffectorKinematicControllerWithNullSpaceTask( EndEffectorKinematicController ) :
    """ 
    Kinematic effector coordinates controller using the Jacobian of the system
    inlcuding secondary control loop v = (q_d - q) * k2 with joint space 
    position target projected on the nullspace
    ------------------------------------------
    r = r_d : reference signal vector  e   x 1
    y = q   : sensor signal vector     dof x 1
    u = dq  : control inputs vector    dof x 1
    t       : time                     1   x 1
    -------------------------------------------
    u = c( y , r , t ) = J# [ (r - r_robot(q)) * k ] + [I -J#J] ((q_d - q) * k2 )

    """
    
    ############################
    def __init__(self, robot, k = 1 , k_null = 1):
        """ """
        
        super().__init__( robot , k )
        
        
        self.gains_null = np.ones( self.dof  ) * k_null
        self.q_d        = np.zeros( self.dof  )
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        #u = np.zeros(self.m) 
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e   = r_desired - r_actual
        q_e = self.q_d  - q    
        
        # Effector space speed 
        dr_r    =   e * self.gains
        dq_null = q_e * self.gains_null
        
        # From effector velocity to joint velocities
        if self.dof == self.e:
            
            dq_r = np.dot( np.linalg.inv( J ) , dr_r )
            
        elif self.dof > self.e:
            
            # Pseudo Inverse
            J_pinv = np.linalg.pinv( J )
            
            # Nullspacr projection Matrix
            Null_p = np.identity( self.dof ) - np.dot(J_pinv,J)
            
            dq_r   = np.dot( J_pinv , dr_r ) + np.dot( Null_p , dq_null )
            
        else:
            
            #TODO
            pass
        
        return dq_r
    


###############################################################################
# Dynamic PID controllers
###############################################################################
        
class JointPID( controller.DynamicController ) :
    """ 
    Linear controller for mechanical system with full state feedback (y=x)
    Independent PID for each DOF
    ---------------------------------------
    r  : reference signal vector  dof x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd + int(r - q) * ki

    """
    
    ############################
    def __init__(self, dof = 1, kp = 1, ki = 0, kd = 0):
        """ """

        # Gains
        self.kp = np.ones( dof  ) * kp
        self.kd = np.ones( dof  ) * kd
        self.ki = np.ones( dof  ) * ki
        
        # Dimensions
        self.k = dof      # ref signal dimension
        self.l = dof      # internal states dimensions
        self.m = dof      # assuming fully-actuated system
        self.p = dof * 2  # y = x ( full state feedback )
        
        super().__init__(self.k, self.l, self.m, self.p)
        
        # Label
        self.name = 'Joint PID Controller'
        
        for i in range(self.l):
            self.internal_state_label[i] = 'Int_e dof ' +str(i)
        
        # DoF
        self.dof = dof
        
        
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        :     self.dof   ]
        dq = x[ self.dof : 2 * self.dof   ]
        
        return [ q , dq ]
        
    
    #############################
    def c( self , z , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        de =     - dq
        ie = z          # Integral of errors are internal states 
        
        # PIDs
        u = e * self.kp + de * self.kd + ie * self.ki
        
        return u
    
    
    #############################
    def b(self, z, y, r, t):
        """ 
        INTERNAL CONTROLLER DYNAMIC
        dz/dt = b( z, y, r, t)
        
        INPUTS
        z  : internal states               l x 1
        y  : sensor signal vector          p x 1
        r  : reference signal vector       k x 1
        t  : time                          1 x 1
        
        OUTPUTS
        d z / dt  : time derivative of internal states        l x 1
        """
        
        dz = np.zeros( self.l )
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        
        # Derivative of internal state is the error
        dz = e
        
        return dz
    


###############################################################################
# End-Effector PID
###############################################################################
        
class EndEffectorPID( controller.DynamicController ) :
    """ 
    Linear controller for mechanical system with full state feedback (y=x)
    Independent PID for each DOF
    ---------------------------------------
    r  : reference signal vector  dof x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (rd - r) * kp + ( -dr ) * kd + int(rd- r) * ki

    """
    
    ############################
    def __init__(self, robot, kp = 1, ki = 0, kd = 0):
        """ """
        
        # Using from model
        self.fwd_kin = robot.forward_kinematic_effector
        self.J       = robot.J
        self.e       = robot.e # nb of effector dof
        
        dof = robot.dof

        # Gains
        self.kp = np.ones( dof  ) * kp
        self.kd = np.ones( dof  ) * kd
        self.ki = np.ones( dof  ) * ki
        
        # Dimensions
        self.k = dof      # ref signal dimension
        self.l = dof      # internal states dimensions
        self.m = dof      # assuming fully-actuated system
        self.p = dof * 2  # y = x ( full state feedback )
        
        super().__init__(self.k, self.l, self.m, self.p)
        
        # Label
        self.name = 'End-Effector PID Controller'
        
        for i in range(self.l):
            self.internal_state_label[i] = 'Int_e axis ' +str(i)
        
        # DoF
        self.dof = dof
        
        
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        :     self.dof   ]
        dq = x[ self.dof : 2 * self.dof   ]
        
        return [ q , dq ]
        
    
    #############################
    def c( self , z , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        dr          = np.dot( J , dq )
        
        # Error
        e  = r_desired - r_actual
        de =           - dr
        ie = z
        
        # Effector space PID
        f = e * self.kp + de * self.kd + ie * self.ki
        
        # From effector force to joint torques
        u = np.dot( J.T , f )
        
        return u
    
    
    #############################
    def b( self, z, y, r, t):
        """ 
        INTERNAL CONTROLLER DYNAMIC
        dz/dt = b( z, y, r, t)
        
        INPUTS
        z  : internal states               l x 1
        y  : sensor signal vector          p x 1
        r  : reference signal vector       k x 1
        t  : time                          1 x 1
        
        OUTPUTS
        d z / dt  : time derivative of internal states        l x 1
        """
        
        dz = np.zeros( self.l )
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        # Derivative of internal state is the error
        dz = e
        
        return dz
    

###############################################################################
# Backup
###############################################################################
        
class old_JointPID( RobotController ) :
    """ 
    Linear controller for mechanical system with full state feedback (y=x)
    Independent PID for each DOF
    ---------------------------------------
    r  : reference signal vector  dof x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd + int(r - q) * ki

    """
    
    ############################
    def __init__(self, dof = 1, kp = 1, ki = 0, kd = 0):
        """ """
        
        super().__init__( dof )
        
        # Label
        self.name = 'Joint PID Controller'
        
        # Gains
        self.kp = np.ones( dof  ) * kp
        self.kd = np.ones( dof  ) * kd
        self.ki = np.ones( dof  ) * ki
        
        # TODO Update this is not a static controller !!!!
        # Integral Memory
        self.dt    = 0.001
        self.e_int = np.zeros( dof )
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTSS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        de =     - dq
        ie = self.e_int + e * self.dt # TODO use dynamic controller class
        
        # PIDs
        u = e * self.kp + de * self.kd + ie * self.ki
        
        # Memory
        self.e_int =  ie # TODO use dynamic controller class
        
        return u
    


###############################################################################
# End-Effector PID
###############################################################################
        
class old_EndEffectorPID( RobotController ) :
    """ 
    PID in effector coordinates, using the Jacobian of the system
    ---------------------------------------
    r  : reference signal vector  e   x 1
    y  : sensor signal vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd + int(r - q) * ki

    """
    
    ############################
    def __init__(self, robot, kp = 1, ki = 0, kd = 0):
        """ """
        
        # Using from model
        self.fwd_kin = robot.forward_kinematic_effector
        self.J       = robot.J
        self.e       = robot.e # nb of effector dof
        
        super().__init__( robot.dof )
        
        # Label
        self.name = 'End-Effector PID Controller'
        
        # Gains
        self.kp = np.ones( self.dof  ) * kp
        self.kd = np.ones( self.dof  ) * kd
        self.ki = np.ones( self.dof  ) * ki
        
        # TODO Update this is not a static controller !!!!
        # Integral Memory
        self.dt    = 0.001
        self.e_int = np.zeros( self.dof )
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        dr          = np.dot( J , dq )
        
        # Error
        e  = r_desired - r_actual
        de =           - dr
        ie = self.e_int + e * self.dt # TODO use dynamic controller class
        
        # Effector space PID
        f = e * self.kp + de * self.kd + ie * self.ki
        
        # From effector force to joint torques
        u = np.dot( J.T , f )
        
        # Memory
        self.e_int =  ie # TODO use dynamic controller class
        
        return u
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    pass