# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: alxgr
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as interpol2D
from scipy.interpolate import RegularGridInterpolator as rgi

from mpl_toolkits.mplot3d import Axes3D

from pyro.control import controller


'''
################################################################################
'''


class ViController(controller.StaticController):
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     m x 1
    u  : control inputs vector    p x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) 

    """
    ############################
    def __init__(self, k, m, p):
        """ """
        
        # Dimensions
        self.k = k   
        self.m = m   
        self.p = p
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Value Iteration Controller'
        
    
    #############################
    def vi_law( self , x ):
        """   """
        u = np.zeros(self.m) # State derivative vector
        return u

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        u = self.vi_law( x )
        
        return u
    


##############################################################################
# Value Iteration x dim = 2, u dim = 1
##############################################################################

class ValueIteration_2D:
    """ Dynamic programming for 2D continous dynamic system, one continuous input u """
    
    ############################
    def __init__(self, grid_sys , cost_function ):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys       = grid_sys.sys     # Base Dynamic system class
        
        # Controller
        self.ctl = ViController( self.sys.n , self.sys.m , self.sys.n)
        
        # Cost function
        self.cf  = cost_function
        
        # Print params
        self.fontsize = 10
        
        
        # Options
        self.uselookuptable = True
        
        
    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """

        self.J             = np.zeros( self.grid_sys.xgriddim , dtype = float )
        self.action_policy = np.zeros( self.grid_sys.xgriddim , dtype = int   )

        self.Jnew          = self.J.copy()
        self.Jplot         = self.J.copy()

        # Initial evaluation
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                
                # Final Cost
                self.J[i,j] = self.cf.h( x )
                        
                
    ###############################
    def compute_step(self):
        """ One step of value iteration """
        
        # Get interpolation of current cost space
        J_interpol = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.J , bbox=[None, None, None, None], kx=1, ky=1,)
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for action in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.actions_input[ action , : ]
                    
                    # Compute next state and validity of the action                    
                    
                    if self.uselookuptable:
                        
                        x_next        = self.grid_sys.x_next[node,action,:]
                        action_isok   = self.grid_sys.action_isok[node,action]
                        
                    else:
                        
                        x_next        = self.sys.f( x , u ) * self.grid_sys.dt + x
                        x_ok          = self.sys.isavalidstate(x_next)
                        u_ok          = self.sys.isavalidinput(x,u)
                        action_isok   = ( u_ok & x_ok )
                    
                    # If the current option is allowable
                    if action_isok:
                        
                        J_next = J_interpol( x_next[0] , x_next[1] )
                        
                        # Cost-to-go of a given action
                        y = self.sys.h(x, u, 0)
                        Q[action] = self.cf.g(x, u, y, 0) * self.grid_sys.dt + J_next[0,0]
                        
                    else:
                        # Not allowable states or inputs/states combinations
                        Q[action] = self.cf.INF
                        
                        
                self.Jnew[i,j]          = Q.min()
                self.action_policy[i,j] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.Jnew[i,j] > (self.cf.INF-1) :
                    self.action_policy[i,j]      = -1
        
        
        # Convergence check        
        delta = self.J - self.Jnew
        j_max     = self.Jnew.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print('Max:',j_max,'Delta max:',delta_max, 'Delta min:',delta_min)
        
        self.J = self.Jnew.copy()
        

    ################################
    def assign_interpol_controller(self):
        """ controller from optimal actions """
        
        # Compute grid of u
        self.u_policy_grid    = []
        
        # for all inputs
        for k in range(self.sys.m):
            self.u_policy_grid.append( np.zeros( self.grid_sys.xgriddim , dtype = float ) )
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                
                # If no action is good
                if ( self.action_policy[i,j] == -1 ):
                    
                    # for all inputs
                    for k in range(self.sys.m):
                        self.u_policy_grid[k][i,j] = 0 
                    
                else:
                    # for all inputs
                    for k in range(self.sys.m):
                        self.u_policy_grid[k][i,j] = self.grid_sys.actions_input[ self.action_policy[i,j] , k ]
        

        # Compute Interpol function
        self.x2u_interpol_functions = []
        
        # for all inputs
        for k in range(self.sys.m):
            self.x2u_interpol_functions.append(
                    interpol2D( self.grid_sys.xd[0] , 
                                self.grid_sys.xd[1] , 
                                self.u_policy_grid[k] , 
                                bbox=[None, None, None, None], 
                                kx=1, ky=1,) )
        
        # Asign Controller
        self.ctl.vi_law = self.vi_law
        
        
        
    ################################
    def vi_law(self, x , t = 0 ):
        """ controller from optimal actions """
        
        u = np.zeros( self.sys.m )
        
        # for all inputs
        for k in range(self.sys.m):
            u[k] = self.x2u_interpol_functions[k]( x[0] , x[1] )
        
        return u
    
    
    
    ################################
    def compute_steps(self, l = 50, plot = False):
        """ compute number of step """
               
        for i in range(l):
            print('Step:',i)
            self.compute_step()
            
            
                
    ################################
    def plot_cost2go(self, maxJ = 1000 ):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        self.Jplot = self.J.copy()
        
        ## Saturation function for cost
        for i in range(self.grid_sys.xgriddim[0]):
            for j in range(self.grid_sys.xgriddim[1]):
                if self.J[i,j] >= maxJ :
                    self.Jplot[i,j] = maxJ
                else:
                    self.Jplot[i,j] = self.J[i,j]
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Cost-to-go')
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = self.fontsize)
        plt.xlabel(xname, fontsize = self.fontsize)
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] ,
                                   self.grid_sys.xd[1] , 
                                   self.Jplot.T,
                                   shading='gouraud')
        
        plt.axis([self.sys.x_lb[0],
                  self.sys.x_ub[0],
                  self.sys.x_lb[1], 
                  self.sys.x_ub[1]])
    
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        
    
    ################################
    def plot_policy(self, i = 0 ):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        policy_plot = self.u_policy_grid[i].copy()
                
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Policy for u[%i]'%i)
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = self.fontsize )
        plt.xlabel(xname, fontsize = self.fontsize )
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] , 
                                   self.grid_sys.xd[1] , 
                                   policy_plot.T,
                                   shading='gouraud')
        
        plt.axis([self.sys.x_lb[0], 
                  self.sys.x_ub[0], 
                  self.sys.x_lb[1], 
                  self.sys.x_ub[1]])
    
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout() 
        
        
    ################################
    def load_data(self, name = 'DP_data'):
        """ Save optimal controller policy and cost to go """
        
        try:

            self.J              = np.load( name + '_J'  + '.npy' )
            self.action_policy  = np.load( name + '_a'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load DP data ' )
        
        
    ################################
    def save_data(self, name='DP_data', prefix=''):
        """ Save optimal controller policy and cost to go """

        np.save(prefix + name + '_J', self.J)
        np.save(prefix + name + '_a', self.action_policy.astype(int))
        
        
        
        
##############################################################################
# Value Iteration generic algo
##############################################################################

class ValueIteration_ND:
    """ Dynamic programming for continuous dynamic system """

    ############################
    def __init__(self, grid_sys, cost_function):

        # Dynamic system
        self.grid_sys = grid_sys  # Discretized Dynamic system class
        self.sys = grid_sys.sys  # Base Dynamic system class

        # initializes nb of dimensions and continuous inputs u
        self.n_dim = self.sys.n
        
        # Controller
        self.ctl = ViController(self.sys.n, self.sys.m, self.sys.n)

        # Cost function
        self.cf = cost_function

        # Print params
        self.fontsize             = 5
        self.figsize              = (4, 3)
        self.dpi                  = 300
        self.plot_max_J           = 10000

        # Options
        self.uselookuptable = True

    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """
        # Initial evaluation

        # J-arrays and action policy arrays
        self.J = np.zeros(self.grid_sys.xgriddim, dtype=float)
        self.action_policy = np.zeros(self.grid_sys.xgriddim, dtype=int)

        self.Jnew = self.J.copy()
        self.Jplot = self.J.copy()

        # For all state nodes
        for node in range(self.grid_sys.nodes_n):
            x = self.grid_sys.nodes_state[node, :]

            # use tuple to get dynamic list of indices
            indices = tuple(self.grid_sys.nodes_index[node, i] for i in range(self.n_dim))

            # Final cost
            self.J[indices] = self.cf.h(x)

        print('J shape:', self.J.shape)

    ###############################
    def compute_step(self):
        """ One step of value iteration """

        # Get interpolation of current cost space
        if self.n_dim == 2:
            J_interpol = interpol2D(self.grid_sys.xd[0], self.grid_sys.xd[1],
                                    self.J, bbox=[None, None, None, None], kx=1, ky=1)
        elif self.n_dim == 3:
            # call function for random shape
            J_interpol = rgi([self.grid_sys.xd[0], self.grid_sys.xd[1], self.grid_sys.xd[2]], self.J)
        else:
            points = tuple(self.grid_sys.xd[i] for i in range(self.n_dim))
            J_interpol = rgi(points, self.J)

        # For all state nodes
        for node in range(self.grid_sys.nodes_n):

            x = self.grid_sys.nodes_state[node, :]

            # use tuple to get dynamic list of indices
            indices = tuple(self.grid_sys.nodes_index[node, i] for i in range(self.n_dim))

            # One steps costs - Q values
            Q = np.zeros(self.grid_sys.actions_n)

            # For all control actions
            for action in range(self.grid_sys.actions_n):

                u = self.grid_sys.actions_input[action, :]

                # Compute next state and validity of the action

                if self.uselookuptable:

                    x_next = self.grid_sys.x_next[node, action, :]
                    action_isok = self.grid_sys.action_isok[node, action]

                else:

                    x_next = self.sys.f(x, u) * self.grid_sys.dt + x
                    x_ok = self.sys.isavalidstate(x_next)
                    u_ok = self.sys.isavalidinput(x, u)
                    action_isok = (u_ok & x_ok)

                # If the current option is allowable
                if action_isok:
                    if self.n_dim == 2:
                        J_next = J_interpol(x_next[0], x_next[1])
                    elif self.n_dim == 3:
                        J_next = J_interpol([x_next[0], x_next[1], x_next[2]])
                    else:
                        J_next = J_interpol([x_next[0], x_next[1], x_next[2], x_next[3]])

                    # Cost-to-go of a given action
                    y = self.sys.h(x, u, 0)
                    if self.n_dim == 2:
                        Q[action] = ( self.cf.g(x, u, y, 0) * self.grid_sys.dt
                                    + J_next[0, 0] )
                    else:
                        Q[action] = ( self.cf.g(x, u, y, 0) * self.grid_sys.dt 
                                     + J_next)

                else:
                    # Not allowable states or inputs/states combinations
                    Q[action] = self.cf.INF

            self.Jnew[indices] = Q.min()
            self.action_policy[indices] = Q.argmin()

            # Impossible situation ( unaceptable situation for any control actions )
            if self.Jnew[indices] > (self.cf.INF - 1):
                self.action_policy[indices] = -1

        # Convergence check
        delta = self.J - self.Jnew
        j_max = self.Jnew.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print('Max:', j_max, 'Delta max:', delta_max, 'Delta min:', delta_min)

        self.J = self.Jnew.copy()

        # TODO: Combine deltas? Check if delta_min or max changes
        return delta_min

    ################################
    def assign_interpol_controller(self):
        """ controller from optimal actions """

        # Compute grid of u
        self.u_policy_grid = []

        # for all inputs
        for k in range(self.sys.m):
            self.u_policy_grid.append(np.zeros(self.grid_sys.xgriddim, dtype=float))

        # For all state nodes
        for node in range(self.grid_sys.nodes_n):

            # use tuple to get dynamic list of indices
            indices = tuple(self.grid_sys.nodes_index[node, i] for i in range(self.n_dim))

            # If no action is good
            if (self.action_policy[indices] == -1):

                # for all inputs
                for k in range(self.sys.m):
                    self.u_policy_grid[k][indices] = 0

            else:
                # for all inputs
                for k in range(self.sys.m):
                    self.u_policy_grid[k][indices] = self.grid_sys.actions_input[self.action_policy[indices], k]

        # Compute Interpol function
        self.interpol_functions = []

        # for all inputs
        for k in range(self.sys.m):
            if self.n_dim == 2:
                self.interpol_functions.append(
                    interpol2D(self.grid_sys.xd[0],
                               self.grid_sys.xd[1],
                               self.u_policy_grid[k],
                               bbox=[None, None, None, None],
                               kx=1, ky=1, ))
            elif self.n_dim == 3:
                self.interpol_functions.append(
                    rgi([self.grid_sys.xd[0], self.grid_sys.xd[1], self.grid_sys.xd[2]], self.u_policy_grid[k]))
            else:
                points = tuple(self.grid_sys.xd[i] for i in range(self.n_dim))
                self.interpol_functions.append(
                    rgi(points,
                                            self.u_policy_grid[k],
                                            method='linear'))

        # Asign Controller
        self.ctl.vi_law = self.vi_law

    ################################
    def vi_law(self, x, t=0):
        """ controller from optimal actions """

        u = np.zeros(self.sys.m)

        for i in range(self.sys.n):
            if x[i] < self.sys.x_lb[i]:
                x[i] = self.sys.x_lb[i]
            if x[i] > self.sys.x_ub[i]:
                x[i] = self.sys.x_ub[i]

        # for all inputs
        for k in range(self.sys.m):
            if self.n_dim == 2:
                u[k] = self.interpol_functions[k](x[0], x[1])
            else:
                u[k] = self.interpol_functions[k](x)

        return u

    ################################
    def compute_steps(self, l=50, plot=False, threshold=1.0e-25, maxJ=1000):
        """ compute number of step """
        step = 0
        print('Step:', step)
        cur_threshold = self.compute_step()
        print('Current threshold', cur_threshold)
        if plot:
            self.plot_dynamic_cost2go()
        # while abs(cur_threshold) > threshold:
        while step < l:
            step = step + 1
            print('Step:', step)
            cur_threshold = self.compute_step()
            print('Current threshold', cur_threshold)
            if plot:
                self.draw_cost2go( step, maxJ )

    ################################
    def plot_dynamic_cost2go(self):
        """ print graphic """
        
        maxJ = self.plot_max_J

        #plt.ion()

        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]

        self.fig_dynamic = plt.figure(figsize= self.figsize, dpi=self.dpi, frameon=True)
        self.fig_dynamic.canvas.set_window_title('Dynamic Cost-to-go')
        self.ax1_dynamic = self.fig_dynamic.add_subplot(1, 1, 1)

        plt.ylabel(yname, fontsize=self.fontsize)
        plt.xlabel(xname, fontsize=self.fontsize)

        plt.axis([self.sys.x_lb[0],
                  self.sys.x_ub[0],
                  self.sys.x_lb[1],
                  self.sys.x_ub[1]])

        self.Jplot = self.J.copy()
        self.create_Jplot()
        
        
        #TODO update next line for deciding axis to plot
        #plot = self.Jplot.T if self.n_dim == 2 else self.Jplot[..., 0].T
        
        if self.n_dim == 2:
            plot = self.Jplot.T
            
        elif self.n_dim == 3:
            plot = self.Jplot[..., 0].T
            
        elif self.n_dim == 4:
            plot = self.Jplot[...,0,0].T
            
        else:
            raise NotImplementedError()
            
        
        self.im1_dynamic = plt.pcolormesh(self.grid_sys.xd[0],
                                  self.grid_sys.xd[1],
                                  plot,
                                  shading='gouraud')
        
        self.step_text_template = 'Number of steps = %i'
        self.time_text = self.ax1_dynamic.text(
                0.05, 0.05, '', transform=self.ax1_dynamic.transAxes, 
                fontsize=self.fontsize )
        
        self.ax1_dynamic.tick_params( labelsize = self.fontsize )

        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()

        #plt.draw()
        plt.pause(0.001)
        
        plt.ion()


    #############################
    def draw_cost2go(self, step, maxJ=1000):
        self.Jplot = self.J.copy()
        self.create_Jplot()
        
        if self.n_dim == 2:
            plot = self.Jplot.T
            
        elif self.n_dim == 3:
            plot = self.Jplot[..., 0].T
            
        elif self.n_dim == 4:
            plot = self.Jplot[...,0,0].T
            
        else:
            raise NotImplementedError()
        
        self.im1_dynamic.set_array(np.ravel(plot))
        self.time_text.set_text(self.step_text_template % ( step ))
        #plt.draw()
        plt.pause(0.001)
        

    ################################
    def create_Jplot(self):
        
        maxJ = self.plot_max_J
        
        ## Saturation function for cost
        if self.n_dim == 2:
            for i in range(self.grid_sys.xgriddim[0]):
                for j in range(self.grid_sys.xgriddim[1]):
                    self.Jplot[i, j] = maxJ if self.J[i, j] >= maxJ else self.J[i, j]
                    
        elif self.n_dim == 3:
            for i in range(self.grid_sys.xgriddim[0]):
                for j in range(self.grid_sys.xgriddim[1]):
                    for k in range(len(self.J[i, j])):
                        self.Jplot[i, j, k] = maxJ if self.J[i, j, k] >= maxJ else self.J[i, j, k]
                        
        elif self.n_dim == 4:
            for i in range(self.grid_sys.xgriddim[0]):
                for j in range(self.grid_sys.xgriddim[1]):
                    for k in range(self.grid_sys.xgriddim[2]):
                        for l in range(self.grid_sys.xgriddim[3]):
                            self.Jplot[i, j, k, l] = maxJ if self.J[i, j, k, l] >= maxJ else self.J[i, j, k, l]
        

    ################################
    def plot_cost2go(self, maxJ=1000):
        """ print graphic """
        
        self.plot_max_J = maxJ

        self.plot_dynamic_cost2go()
        #self.draw_cost2go()


    ################################
    def plot_policy(self, i=0):
        """ print graphic """

        plt.ion()

        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]

        policy_plot = self.u_policy_grid[i].copy()

        self.fig1 = plt.figure(figsize=(4, 4), dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Policy for u[%i]' % i)
        self.ax1 = self.fig1.add_subplot(1, 1, 1)

        #plot = policy_plot.T if self.n_dim == 2 else policy_plot[..., 0].T
        
        if self.n_dim == 2:
            plot = policy_plot.T
            
        elif self.n_dim == 3:
            plot = policy_plot[..., 0].T
            
        elif self.n_dim == 4:
            plot = policy_plot[...,0,0].T
            
        else:
            raise NotImplementedError()
        
        plt.ylabel(yname, fontsize=self.fontsize)
        plt.xlabel(xname, fontsize=self.fontsize)
        self.im1 = plt.pcolormesh(self.grid_sys.xd[0],
                                  self.grid_sys.xd[1],
                                  plot,
                                  shading='gouraud')

        plt.axis([self.sys.x_lb[0],
                  self.sys.x_ub[0],
                  self.sys.x_lb[1],
                  self.sys.x_ub[1]])

        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()

        plt.draw()
        plt.pause(0.001)

        ################################

    def plot_3D_policy(self, i=0):
        """ print graphic """

        plt.ion()

        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]

        policy_plot = self.u_policy_grid[i].copy()
        print(policy_plot.shape)

        self.fig1 = plt.figure()
        self.fig1.canvas.set_window_title('Policy for u[%i]' % i)
        self.ax1 = self.fig1.gca(projection='3d')

        plot = policy_plot.T if self.n_dim == 2 else policy_plot[..., 0].T
        plt.ylabel(yname, fontsize=self.fontsize)
        plt.xlabel(xname, fontsize=self.fontsize)
        X = plot[:, 0]
        Y = plot[:, 1]
        Z = plot[:, 2]
        self.ax1.plot_trisurf(X, Y, Z)

        plt.axis([self.sys.x_lb[0],
              self.sys.x_ub[0],
              self.sys.x_lb[1],
              self.sys.x_ub[1]])

        # plt.colorbar()

        plt.draw()
        plt.pause(1)

        ################################

    def load_data(self, name='DP_data', prefix=''):
        """ Save optimal controller policy and cost to go """

        try:
            self.J = np.load(prefix + name + '_J' + '.npy')
            self.action_policy = np.load(prefix + name + '_a' + '.npy').astype(int)
            print('File successfully loaded')

        except IOError:
            type, value, traceback = sys.exc_info()
            print('Error opening %s: %s' % (value.filename, value.strerror))

        # returns filled array to signal that the trajectory has been loaded
        # used in Slash library
        return [1]

    ################################
    def save_data(self, name='DP_data', prefix=''):
        """ Save optimal controller policy and cost to go """

        print('Final J', self.J)
        print('Final policy', self.action_policy)

        np.save(prefix + name + '_J', self.J)
        np.save(prefix + name + '_a', self.action_policy.astype(int))
