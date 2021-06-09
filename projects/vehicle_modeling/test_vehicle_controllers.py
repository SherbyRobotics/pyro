#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:54:02 2019

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control import controller
###############################################################################


class kinematicInputs( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ----------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector         k x 1
    t  : time                          1 x 1
    ----------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = 3   
        self.m = 2   
        self.p = 3
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
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
        max_steer = 0.15
        max_speed = 5.0000
        u = np.zeros(self.m) # State derivative vector
        
        if t<1:
            u[0] = 0
            #u[0] = max_steer
            u[1] = max_speed
            #u[1] = 0
        elif(t>=1 and t<=3):
            u[0] = float(max_steer/2.0000*(t-1.0000))
            #u[0] = max_steer
            #u[1] = float(max_speed/40.0000*(t-10.0000))
            u[1] = max_speed
        elif(t>3 and t<=90):
            #u[0] = float(max_steer-max_steer/40.0000*(t-50.0000))
            u[0] = max_steer
            u[1] = max_speed
        else:
            #u[0] = 0
            u[0] = max_steer
            u[1] = max_speed
        
        
        return u
    


class dynLongVelInputs( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = 5   
        self.m = 2   
        self.p = 5
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
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
        max_steer = 0.15
        max_speed = 15.0000
        u = np.zeros(self.m) # State derivative vector
        
        if t<1:
            u[0] = 0
            #u[0] = max_steer
            u[1] = max_speed
            #u[1] = 0
        elif(t>=1 and t<=3):
            u[0] = float(max_steer/2.0000*(t-1.0000))
            #u[0] = max_steer
            #u[1] = float(max_speed/40.0000*(t-10.0000))
            u[1] = max_speed
        elif(t>3 and t<=90):
            #u[0] = float(max_steer-max_steer/40.0000*(t-50.0000))
            u[0] = max_steer
            u[1] = max_speed
        else:
            #u[0] = 0
            u[0] = max_steer
            u[1] = max_speed
        
        
        return u

class dynLongForcesInputs( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = 6 
        self.m = 3   
        self.p = 6
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
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
        
        u = np.zeros(self.m) # State derivative vector
        steer_max = 0.3
        
        if (t>=0 and t<25):
            u[0] = steer_max/25*t
            u[2] = 1500
        elif(t>=25 and t<50):
            u[0] = steer_max
            #u[0] = steer_max-steer_max/25*(t-25)
            u[2] = 0
        else:
            #u[0] = 0
            u[0] = steer_max
            u[2] = 0
        u[1] = 0
        
        
        return u
    
class fullDynTorqueInputs( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = 8  
        self.m = 3   
        self.p = 8
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
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
        
        u = np.zeros(self.m) # State derivative vector
        #steer_max = 0.3
        torque_max = 1000
        if t<10:
            u[0] = 0
            u[2] = 1000
        elif(t>=10 and t<20):
            u[0] = 0
            #u[0] = steer_max/40*(t-10)
            u[2] = 1000-1000/10.000*(t-10)
        elif(t>=20 and t<25):
            u[0] = 0
            u[2] = 0
        elif(t>=25 and t<35):
            u[0] = 0
            #u[0] = steer_max
            #u[2] = 0
            u[2] = -torque_max/10.0000*(t-25)
        else:
            u[0] = 0
            u[2] = -torque_max
        u[1] = 0
        
        
        return u
    
class fullDynVoltInputs( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = 7  
        self.m = 2   
        self.p = 7
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
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
        
        u = np.zeros(self.m) # State derivative vector
        #steer_max = 0.3
        volt_max = 8
        steer_max = 0.3
        if t<10:
            u[0] = 0
            #u[1] = 0
            u[1] = 5
        elif(t>=10 and t<20):
            u[0] = steer_max/10.000*(t-10)
            #u[1] = volt_max/10.000*(t-10)
            u[1] = 5
        elif(t>=20 and t<25):
            u[0] = steer_max
            #u[1] = volt_max
            u[1] = 5
        elif(t>=25 and t<35):
            u[0] = steer_max-steer_max/10.0000*(t-25)
            #u[1] = volt_max-volt_max/10.0000*(t-25)
            u[1] = 5
        else:
            u[0] = 0
            #u[1] = -volt_max
            u[1] = 5

        
        
        return u