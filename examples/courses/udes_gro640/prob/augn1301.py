#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

import numpy as np
from scipy.optimize import fsolve
from pyro.control  import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    
    T = np.zeros((4,4))
    
    ###################
    # Votre code ici
    ###################
    T = np.array([
              [np.cos(theta), -1*np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
              [np.sin(theta), np.cos(theta)*np.cos(alpha), -1*np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
              [0, np.sin(alpha), np.cos(alpha), d],
              [0, 0, 0, 1]
              ])
    
    return T



def dhs2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    
    WTT = np.zeros((4,4))
    
    ###################
    # Votre code ici
    ###################
    
    n = len(r)

    WTT = np.identity(4)

    for i in range(n):
        WTT = WTT @ dh2T(r[i], d[i], theta[i], alpha[i])

    return WTT


def f(q):
    """
    

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position

    """
    r = np.zeros((3,1))
    
    ###################
    # Votre code ici
    ###################
   
    r_dh = np.array([0.033, 0.155, 0.135, 0, 0, 0]) 
    d_dh = np.array([0.147, 0, 0, 0, 0.217, q[5]])
    theta_dh = np.array([q[0], q[1]-np.pi/2, q[2], np.pi/2+q[3], q[4], 0])
    alpha_dh = np.array([-np.pi/2, 0, 0, np.pi/2, np.pi/2, 0])
    
    WTT = dhs2T( r_dh , d_dh , theta_dh, alpha_dh )
    
    r[0] = WTT[0,3]
    r[1] = WTT[1,3]
    r[2] = WTT[2,3]
    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        ################
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        
        lamda = 0.5
        
        dq = np.linalg.inv(J.T @ J + lamda**2*np.identity(3)) @ J.T @ e

        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( robotcontrollers.RobotController ) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        self.state = 0
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref
        #f_e = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
            
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        
        u = np.zeros(self.m)  # place-holder de bonne dimension
               
        k = np.diag([100,100,100])
        b = np.diag([50,50,50])
        
        dr = J @ dq
        
        if self.state == 0: # Approche
            r_d = np.array([0.25,0.25,1])
            fe = k @ (r_d - r) - (b @ dr)
            if np.isclose(r_d, r, atol=0.01).all():
                self.state = 1
        
        elif self.state == 1: # Aligne au trou
            r_d = np.array([0.25,0.25,0.39])
            fe = k @ (r_d - r) - (b @ dr)
            if np.isclose(0.39 ,r[2], atol=0.01):
                self.state = 2
            
        elif self.state == 2: # Percage
            fe = np.array([0,0,-200])
            if np.isclose(0.2 ,r[2], atol=0.01):
                self.state = 3
            
        elif self.state == 3: # Retrait
            r_d = np.array([0.25,0.25,1])
            fe = k @ (r_d - r) - (b @ dr)
            
        u = J.T @ fe + g
        
        return u
        
    
###################
# Part 4
###################
        
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################

    t = np.linspace(0, t_f, l)
    
    # Profil temporelpolynomial d'orde 3 
    s = 3/t_f**2 * t**2 - 2/t_f**3 * t**3
    ds = 6/t_f**2 * t - 6/t_f**3 * t**3
    dds = 6/t_f**2 - 12/t_f**3 * t
    
    # Calcul de r, dr et ddr avec s
    for i in range(l):
        r[:,i] = r_0 + s[i]*(r_f - r_0)
        dr[:,i] = ds[i]*(r_f - r_0)
        ddr[:,i] = dds[i]*(r_f - r_0)
    return r, dr, ddr


def dJ_calc(q, dq, manipulator, n):
    # Calculating the derivaive of the jacobian
    dJ = np.zeros((n,n))
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3
    
    [c1,s1,c2,s2,c3,s3,c23,s23] = manipulator.trig( q )
    dq1 = dq[0]
    dq2 = dq[1]
    dq3 = dq[2]

    dJ[0,0] =  -c1*(l3*c23 + l2*c2) * dq1 + -s1*(-l3*s23 - l2*s2) * dq2 + -s1*(-l3*s23) * dq3      
    dJ[0,1] =  s1*(l3*s23 + l2*s2) * dq1 + -c1*(l3*c23 + l2*c2) * dq2 + -c1*(l3*c23) * dq3
    dJ[0,2] =  (l3*s23*s1) * dq1 + (-l3*c23*c1) * dq2 + (-l3*c23*c1) * dq3
    
    dJ[1,0] =   -s1*(l3*c23 + l2*c2) * dq1 + c1*(-l3*s23 - l2*s2) * dq2 + c1*(-l3*s23) * dq3
    dJ[1,1] =  -c1*(l3*s23 + l2*s2) * dq1 + -s1*(l3*c23 + l2*c2) * dq2 + -s1*(l3*c23) * dq3
    dJ[1,2] =  (-l3*s23*c1) * dq1 + (-l3*c23*s1) * dq2 + (-l3*c23*s1) * dq3
    
    dJ[2,0] =  0
    dJ[2,1] =  (-l3*s23 - l2*s2) * dq2 + (-l3*s23) * dq3
    dJ[2,2] =  (-l3*s23) * dq2 + (-l3*s23) * dq3
    
    return dJ

def sys_equ(q, r, manipulator):
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3
    
    # Equations cinematique direct
    eq_x = ((l2*np.cos(q[1]) + l3*np.cos(q[1]+q[2]))*np.cos(q[0])) - r[0]
    eq_y = ((l2*np.cos(q[1]) + l3*np.cos(q[1]+q[2]))*np.sin(q[0])) - r[1]
    eq_z = (l1 + l2*np.sin(q[1]) + l3*np.sin(q[1]+q[2])) - r[2]
    
    eq = np.array([eq_x,eq_y,eq_z])
    return eq
    
def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ################################# 
    J = np.zeros((n,n))
    dJ = np.zeros((n,n))
    
    # Condition initial pour le solver
    q[:,0] = np.array([0, (23/60)*np.pi, -(7/36)*np.pi])
    dq[:,0] = np.array([0, 0, 0])
    
    for i in range(l-1):
        # Resolution cinemtaique inverse
        q[:,i+1] = fsolve(sys_equ, q[:,i], args = (r[:,i+1], manipulator), )
        
        # Calcul matrice Jacobienne
        J = manipulator.J(q[:,i+1])
        
        # Calcul dq
        dq[:,i+1] = np.linalg.inv(J) @ dr[:,i+1]
        
        # Calcul derive matrice jacobienne
        dJ = dJ_calc(q[:,i], dq[:,i], manipulator, n)
        
        #Calcul de ddq
        ddq[:,i] = np.linalg.inv(J) @ (ddr[:,i] - dJ @ dq[:,i])

    return q, dq, ddq

def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    for i in range(l):
        H = manipulator.H(q[:,i])
        G = manipulator.G(q[:,i])
        C = manipulator.C(q[:,i],dq[:,i])
        B = manipulator.B(q[:,i])
        
        tau[:,i] = np.linalg.inv(B) @ (H @ ddq[:,i] + C @ dq[:,i] + G)
    
    return tau