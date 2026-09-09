#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:48:32 2022

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from scipy.interpolate import RectBivariateSpline as interpol2D
from scipy.interpolate import RegularGridInterpolator as rgi



from pyro.control  import controller


###############################################################################
### DP controllers
###############################################################################

class LookUpTableController( controller.StaticController ):

    ############################
    def __init__(self, grid_sys , pi ):
        """
        Pyro controller based on a discretized lookpup table of control inputs

        Parameters
        ----------
        grid_sys : pyro GridDynamicSystem class
            A discretized dynamic system
        pi : numpy array, dim =  self.grid_sys.nodes_n , dtype = int
            A list of action index for each node id
        """
        
        if grid_sys.nodes_n != pi.size:
            raise ValueError("Grid size does not match optimal action table size")
        
        k = 1                   # Ref signal dim
        m = grid_sys.sys.m      # control input signal dim
        p = grid_sys.sys.n      # output signal dim (state feedback)
        
        super().__init__(k, m, p)
        
        # Grid sys
        self.grid_sys = grid_sys
        
        # Table of actions
        self.pi = pi
        
        # Label
        self.name = 'Tabular Controller'
        
        # Interpolation Options
        self.interpol_method = []
        
        for k in range(self.m):
            
            # options can be changed for each control input axis
            self.interpol_method.append('linear') # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
            
        self.compute_interpol_functions()
        
    
    #############################
    def compute_interpol_functions( self  ):
        """  """
        
        self.u_interpol = [] 
        
        for k in range(self.m):
            
            u_k      = self.grid_sys.get_input_from_policy( self.pi, k)
            self.u_interpol.append( self.grid_sys.compute_interpolation_function( u_k , 
                                                                                 self.interpol_method[k],
                                                                                 bounds_error = False   , 
                                                                                 fill_value = 0  ) )
        
    
    #############################
    def lookup_table_selection( self , x ):
        """  select the optimal u given actual cost map """
        
        x = np.asarray( x ).ravel()
        u = np.zeros( self.m )
        
        for k in range(self.m):
            
            u[k] = self.u_interpol[k]( x.reshape(1, -1) )[0]
            
        return u
    

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        
        x = y
        
        u = self.lookup_table_selection( x )
        
        return u
    
    

###############################################################################
### DP Algo
###############################################################################

class DynamicProgramming:
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0 ):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys       = grid_sys.sys     # Base Dynamic system class
        
        # Cost function
        self.cf  = cost_function
        self.tf  = final_time
        
        # Options
        self.alpha                = 1.0     # exponential forgetting factor
        self.interpol_method      ='linear' # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        self.save_time_history    = True
        
        # Memory Variables
        self.t = self.tf   # time of the computed step (useful for time-varying system)
        self.k = 0         # Number of computed steps
        
        
        # Start time (needed to plot elapsed computation time)
        self.start_time = time.time()
        
        # Final cost
        self.evaluate_terminal_cost()
        
        # 
        if self.save_time_history:

            self.t_list  = []
            self.J_list  = []
            self.pi_list = []
            
            # Value at t = t_f
            self.J_list.append(  self.J  )
            self.t_list.append(  self.tf )
            self.pi_list.append( self.pi )
        
        
    ##############################
    def evaluate_terminal_cost(self):
        """ initialize cost-to-go and policy """

        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )

        # Initial cost-to-go evaluation       
        for s in range( self.grid_sys.nodes_n ):  
            
                xf = self.grid_sys.state_from_node_id[ s , : ]
                
                # Final Cost of all states
                self.J[ s ] = self.cf.h( xf , self.tf )
                
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k      = self.k + 1                  # index backward in time
        self.t      = self.t - self.grid_sys.dt   # time
        self.J_next = self.J
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_interpolation_function( self.J_next               , 
                                                                        self.interpol_method      , 
                                                                        bounds_error = False      , 
                                                                        fill_value = 0  )
                        
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.input_from_action_id[ a , : ]                  
                        
                    # If action is in allowable set
                    if self.sys.isavalidinput( x , u ):
                        
                        # Forward dynamics 
                        x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x
                        
                        # if the next state is not out-of-bound
                        if self.sys.isavalidstate(x_next):

                            # Estimated (interpolation) cost to go of arrival x_next state
                            J_next = self.J_interpol( x_next )
                            
                            # Cost-to-go of a given action
                            Q[ a ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt + self.alpha * J_next
                            
                        else:
                            
                            # Out of bound terminal cost
                            Q[ a ] = self.cf.INF # TODO add option to customize this
                        
                    else:
                        
                        # Invalide control input at this state
                        Q[ a ] = self.cf.INF
                        
                self.J[ s ]  = Q.min()
                self.pi[ s ] = Q.argmin()
                    
    
    ###############################
    def finalize_backward_step(self):
        """ One step of value iteration """
        
        # Computation time
        elapsed_time = time.time() - self.start_time
        
        # Convergence check        
        delta     = self.J - self.J_next
        j_max     = self.J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        
        print('%d t:%.2f Elasped:%.2f max: %.2f dmax:%.2f dmin:%.2f' % (self.k,self.t,elapsed_time,j_max,delta_max,delta_min) )
        
        # List in memory
        if self.save_time_history:
            self.J_list.append(  self.J  )
            self.t_list.append(  self.t  )
            self.pi_list.append( self.pi )
            
        # return largest J change for usage as stoping criteria
        return abs(np.array([delta_max,delta_min])).max() 

    
    ################################
    def compute_steps(self, n = 50 , animate_cost2go = False , animate_policy = False , k = 0 ):
        """ compute number of step """
        
        print('\nComputing %d backward DP iterations:'%n)
        print('-----------------------------------------')
        
        if animate_cost2go: self.plot_cost2go()
        if animate_policy:  self.plot_policy( k )
               
        for i in range(n):
            self.initialize_backward_step()
            self.compute_backward_step()
            self.finalize_backward_step()
            if animate_cost2go: self.update_cost2go_plot()
            if animate_policy:  self.update_policy_plot( k )
            
    
    ################################
    def solve_bellman_equation(self, tol = 0.1 , animate_cost2go = False , animate_policy = False , k = 0 ):
        """ 
        Value iteration algorithm
        --------------------------
        
        Do Dp backward iterations until changes to J are under the tolerance 
        
        Note: self.alpha should be smaller then 1 to garantee convergence
        
        
        """
        
        print('\nComputing backward DP iterations until dJ<%2.2f:'%tol)
        print('---------------------------------------------------------')
        
        #self.plot_cost2go()
        #self.plot_policy( k )
        if animate_cost2go: self.plot_cost2go()
        if animate_policy: self.plot_policy( k )
        
        delta = self.cf.INF
        
        while (delta>tol):
            self.initialize_backward_step()
            self.compute_backward_step()
            delta = self.finalize_backward_step()
            #self.update_cost2go_plot()
            #self.update_policy_plot( k )
            if animate_cost2go: self.update_cost2go_plot()
            if animate_policy: self.update_policy_plot( k )
            
        print('\nBellman equation solved!' )
        
        
    ################################
    ### Data tools
    ################################
    
    ################################
    def clean_infeasible_set(self , tol = 1):
        """
        Set default policy and cost2go to cf.INF for state for  which it is unavoidable
        that they will reach unallowable regions

        """
        
        default_action = self.grid_sys.get_nearest_action_id_from_input( self.sys.ubar )
        
        infeasible_node_IDs = self.J > ( self.cf.INF - tol )
        
        self.J[  infeasible_node_IDs ] = self.cf.INF
        self.pi[ infeasible_node_IDs ] = default_action
        
        
    ################################
    ### Print quick shorcuts
    ################################
            
            
    ################################
    def plot_cost2go(self , jmax = None , i = 0 , j = 1 , show = True ):
        
        if jmax == None: jmax = self.cf.INF
               
        fig, ax, pcm = self.grid_sys.plot_grid_value( self.J , 'Cost-to-go' , i , j , jmax , 0 )
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.cost2go_fig = [fig, ax, pcm, text, i , j ]
        
        plt.ion()
        if show: plt.pause( 0.001 )
        
        
    ################################
    def update_cost2go_plot(self, show = True ):
        
        J_grid = self.grid_sys.get_grid_from_array( self.J )
        
        J_2d = self.grid_sys.get_2D_slice_of_grid( J_grid , self.cost2go_fig[4] , self.cost2go_fig[5] )
               
        self.cost2go_fig[2].set_array( np.ravel( J_2d.T ) )
        self.cost2go_fig[3].set_text('Optimal cost2go at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
        
    
    ################################
    def plot_policy(self , k = 0 , i = 0 , j = 1 , show = True ):
               
        fig, ax, pcm = self.grid_sys.plot_control_input_from_policy( self.pi , k , i , j)
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.policy_fig = [fig, ax, pcm, text, k , i , j ]
        
        plt.ion()
        if show: plt.pause( 0.001 )
        
        
    ################################
    def update_policy_plot(self, show = True  ):
        
        uk    = self.grid_sys.get_input_from_policy( self.pi, self.policy_fig[4] )
        uk_nd = self.grid_sys.get_grid_from_array( uk ) 
        uk_2d = self.grid_sys.get_2D_slice_of_grid( uk_nd , self.policy_fig[5] , self.policy_fig[6] )
               
        self.policy_fig[2].set_array( np.ravel( uk_2d.T ) )
        self.policy_fig[3].set_text('Optimal policy at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
        
        
    ################################
    def animate_cost2go(self, i = 0 , j = 1 , jmax = None , show = True , save = False ,  file_name = 'cost2go_animation'):
        
        self.J  = self.J_list[0]
        #self.pi = self.pi_list[0]
        self.t  = self.t_list[0]
        self.clean_infeasible_set()
        
        self.plot_cost2go( jmax = jmax , i = i , j = j , show = False  )

        self.ani = animation.FuncAnimation( self.cost2go_fig[0], self.__animate_cost2go, 
                                                len(self.J_list), interval = 20 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.cost2go_fig[0].show()
        
        return self.ani
        
    
    ################################
    def __animate_cost2go(self , i ):
        
        self.J  = self.J_list[i]
        #self.pi = self.pi_list[i]
        self.t  = self.t_list[i]
        self.clean_infeasible_set()
        self.update_cost2go_plot( show = False )
        
        
    ################################
    def animate_policy(self , k = 0 , i = 0 , j = 1 , show = True , save = False , file_name = 'policy_animation'):
        
        self.J  = self.J_list[1]
        self.pi = self.pi_list[1]
        self.t  = self.t_list[1]
        self.clean_infeasible_set()
        self.plot_policy( k = k , i = i , j = j , show = False )

        self.ani = animation.FuncAnimation( self.policy_fig[0], self.__animate_policy, 
                                                len(self.pi_list)-1, interval = 20 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.policy_fig[0].show()
        
        return self.ani
        
    
    ################################
    def __animate_policy(self , i ):
        
        self.J  = self.J_list[i+1]
        self.pi = self.pi_list[i+1]
        self.t  = self.t_list[i+1]
        self.clean_infeasible_set()
        self.update_policy_plot( show = False )
        
    
    ################################
    def plot_cost2go_3D(self , jmax = None , i = 0 , j = 1 , show = True ):
        
        if jmax == None: jmax = self.cf.INF
               
        fig, ax, surf = self.grid_sys.plot_grid_value_3D( self.J , None , 'Cost-to-go' , i , j , jmax , 0)
        
        text = ax.text(0.05, 0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.cost2go_3D_fig = [fig, ax, surf, text, i , j ]
        
    
    ################################
    ### Quick utility shorcuts
    ################################
    
    ################################
    def get_lookup_table_controller(self):
        """ Create a pyro controller object based on the latest policy """
        
        ctl = LookUpTableController( self.grid_sys, self.pi )
        
        return ctl
        
        
    ################################
    def save_latest(self, name = 'test_data'):
        """ save cost2go and policy of the latest iteration (further back in time) """
        
        np.save(name + '_J_inf', self.J_next)
        np.save(name + '_pi_inf', self.pi.astype(int) )
        
    
    ################################
    def load_J_next(self, name = 'test_data'):
        """ Load J_next from file """
        
        try:

            self.J_next = np.load( name + '_J_inf'   + '.npy' )
            #self.pi     = np.load( name + '_pi_inf'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load J_next ' )
            

                    
###############################################################################
    
class DynamicProgrammingWithLookUpTable( DynamicProgramming ):
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0 ):
        
        DynamicProgramming.__init__(self, grid_sys, cost_function, final_time)
        
        self.compute_cost_lookuptable()
    
    
    ###############################
    def compute_cost_lookuptable(self):
        """ One step of value iteration """
        
        start_time = time.time()
        print('Computing g(x,u,t) look-up table..  ', end = '')
        
        self.G = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    # If action is in allowable set
                    if self.grid_sys.action_isok[s,a]:
                        
                        # if the next state is not out-of-bound
                        if self.grid_sys.x_next_isok[s,a]:
                            
                            u = self.grid_sys.input_from_action_id[ a , : ]  
                            
                            self.G[ s , a ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                        
                        else:
                            # Out of bound cost (J_interpol return 0 in this case)
                            self.G[ s , a ] = self.cf.INF
                    
                    else:
                        # Not allowable input at this state
                        self.G[ s , a ] = self.cf.INF
        
        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        # Computing the J_next of all x_next in the look-up table
        self.Jx_next = self.J_interpol( self.grid_sys.x_next_table )
        
        # Matrix version of computing all Q values
        self.Q       = self.G + self.alpha * self.Jx_next
                        
        self.J  = self.Q.min( axis = 1 )
        self.pi = self.Q.argmin( axis = 1 )
                

                    
                    

###############################################################################

class DynamicProgramming2DRectBivariateSpline( DynamicProgrammingWithLookUpTable ):
    """ Dynamic programming on a grid sys """
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k      = self.k + 1                  # index backward in time
        self.t      = self.t - self.grid_sys.dt   # time
        self.J_next = self.J
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_bivariatespline_2D_interpolation_function( self.J_next , kx=3, ky=3)
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        # Computing the J_next of all x_next in the look-up table
        X            = self.grid_sys.x_next_table
        Jx_next_flat = self.J_interpol( X[:,:,0].flatten() , X[:,:,1].flatten() , grid = False )
        Jx_next      = Jx_next_flat.reshape( (self.grid_sys.nodes_n , self.grid_sys.actions_n ) )
        
        # Matrix version of computing all Q values
        self.Q       = self.G + self.alpha * Jx_next
                        
        self.J  = self.Q.min( axis = 1 )
        self.pi = self.Q.argmin( axis = 1 )
        
        


###############################################################################
### Policy Evaluation
###############################################################################

class PolicyEvaluator( DynamicProgramming ):
    """ Evaluate the cost2o of a given control law """
    
    ############################
    def __init__(self, ctl , grid_sys , cost_function , final_time = 0 ):
        
        
        DynamicProgramming.__init__(self, grid_sys, cost_function, final_time )
        
        self.ctl = ctl
        
        # Evaluate policy (control law on the grid)
                        
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                ######################################
                # Action taken by the controller
                r = self.ctl.rbar
                u = self.ctl.c( x , r , self.t )   
                ######################################
                    
                # If action is in allowable set
                if self.sys.isavalidinput( x , u ):
                    
                    # Forward dynamics 
                    x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x
                    
                    # if the next state is not out-of-bound
                    if self.sys.isavalidstate(x_next):

                        # Estimated (interpolation) cost to go of arrival x_next state
                        J_next = self.J_interpol( x_next )
                        
                        # Cost-to-go of a given action
                        Q = self.cf.g( x , u , self.t ) * self.grid_sys.dt + self.alpha * J_next
                        
                    else:
                        
                        # Out of bound terminal cost
                        Q = self.cf.INF # TODO add option to customize this
                    
                else:
                    
                    # Invalide control input at this state
                    Q = self.cf.INF
                        
                self.J[ s ]  = Q
                
                

###############################################################################

class PolicyEvaluatorWithLookUpTable( PolicyEvaluator ):
    """ Evaluate the cost2o of a given control law """
    
    ############################
    def __init__(self, ctl , grid_sys , cost_function , final_time = 0 ):
        
        PolicyEvaluator.__init__(self, ctl , grid_sys, cost_function, final_time)
        
        self.compute_lookuptable()
    
    
    ###############################
    def compute_lookuptable(self):
        """ One step of value iteration """
        
        start_time = time.time()
        print('Computing g(x,u,t) and X-next look-up table..  ', end = '')
        
        self.x_next_table = np.zeros( ( self.grid_sys.nodes_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        self.G            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                ######################################
                # Action taken by the controller
                r = self.ctl.rbar
                u = self.ctl.c( x , r , self.t )   
                ######################################
                
                # Forward dynamics 
                x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x
                
                # Save to llokup table
                self.x_next_table[s,:] = x_next
                
                # If action is in allowable set
                if self.sys.isavalidinput( x , u ):
                    
                    # if the next state is not out-of-bound
                    if self.sys.isavalidstate(x_next):
                        
                        self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                    
                    else:
                        # Out of bound cost (J_interpol return 0 in this case)
                        self.G[ s ] = self.cf.INF
                
                else:
                    # Not allowable input at this state
                    self.G[ s ] = self.cf.INF
        
        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.J       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        
        # Computing the J_next of all x_next in the look-up table
        self.Jx_next = self.J_interpol( self.x_next_table )
        
        # Matrix version of computing all Q values
        self.J       = self.G + self.alpha * self.Jx_next


            


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """


    from pyro.dynamic  import pendulum
    from pyro.planning import discretizer
    from pyro.analysis import costfunction

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 300

    # DP algo
    dp = DynamicProgrammingWithLookUpTable(grid_sys, qcf)
    
    dp.solve_bellman_equation( tol = 1.0 )
    
    dp.plot_cost2go()
    dp.plot_cost2go_3D()
    dp.plot_policy()

    