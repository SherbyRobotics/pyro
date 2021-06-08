# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Embed font type in PDF
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
        

###############################################################################
# Phase Plot Object for phase plane analysis
###############################################################################
        
class PhasePlot:
    """ 
    Continous dynamic system phase plot 
    ---------------------------------------------------
    x_axis : index of state to display as x axis
    y_axis : index of state to display as y axis
    
    """
    ############################
    def __init__(self, ContinuousDynamicSystem , x_axis = 0 ,  y_axis = 1):
        
        # System
        self.cds  = ContinuousDynamicSystem
        self.f    = self.cds.f      # dynamic function
        self.xbar = np.copy( self.cds.xbar )  # default state
        self.ubar = np.copy( self.cds.ubar ) # default input
        self.t    = 0               # default time
        
        # Grid
        self.x_axis = x_axis  # Index of state to plot on x axis
        self.y_axis = y_axis  # Index of state to plot on y axis
               
        self.x_axis_min = self.cds.x_lb[ self.x_axis ]
        self.x_axis_max = self.cds.x_ub[ self.x_axis ]
        self.y_axis_min = self.cds.x_lb[ self.y_axis ]
        self.y_axis_max = self.cds.x_ub[ self.y_axis ]
        
        self.x_axis_n = 21
        self.y_axis_n = 21 
        
        # Plotting params
        self.color      = 'b'        
        self.figsize    = (3, 2)
        self.dpi        = 300
        self.linewidth  = 0.005
        self.streamplot = False
        self.arrowstyle = '->'
        self.headlength = 4.5
        self.fontsize   = 5
        
        
    ###########################################################################
    def compute_grid(self):
        
        x = np.linspace( self.x_axis_min , self.x_axis_max , self.x_axis_n )
        y = np.linspace( self.y_axis_min , self.y_axis_max , self.y_axis_n )
        
        self.X, self.Y = np.meshgrid( x, y)
            
        
    ###########################################################################
    def compute_vector_field(self):
        
        self.v = np.zeros(self.X.shape)
        self.w = np.zeros(self.Y.shape)
        
        for i in range(self.x_axis_n):
            for j in range(self.y_axis_n):
                
                # Actual states
                x  = np.copy( self.xbar )   # default value for all states
                x[ self.x_axis ] = self.X[i, j]
                x[ self.y_axis ] = self.Y[i, j]
                
                # States derivative open loop
                dx = self.f( x , self.ubar , self.t ) 
                
                # Assign vector components
                self.v[i,j] = dx[ self.x_axis ]
                self.w[i,j] = dx[ self.y_axis ]
                
                       
    ###########################################################################
    def plot_init(self):
        
        
        self.phasefig = plt.figure( figsize = self.figsize , dpi = self.dpi,
                                   frameon=True)
        self.phasefig.canvas.manager.set_window_title('Phase plane of ' + 
                                                self.cds.name )
        
        
    ###########################################################################
    def plot_vector_field(self):
        
        try:
            self.ax = self.phasefig.axes[0]
        except IndexError:
            self.ax = self.phasefig.add_subplot(111, autoscale_on=False )

        if self.streamplot:
            self.ax.streamplot( self.X, self.Y, self.v, self.w, 
                                    color      =self.color,  
                                    linewidth  = self.linewidth, 
                                    arrowstyle = self.arrowstyle, 
                                    arrowsize  = self.headlength )
        else:
            self.ax.quiver( self.X, self.Y, self.v, self.w, 
                                color     = self.color,  
                                linewidth = self.linewidth)
                                #, headlength = self.headlength )
            
        
    ###########################################################################
    def plot_finish(self):
        
        self.ax.set_xlabel(self.cds.state_label[ self.x_axis ] + ' ' + 
                           self.cds.state_units[ self.x_axis ] , 
                           fontsize = self.fontsize)
        self.ax.set_ylabel(self.cds.state_label[ self.y_axis ] + ' ' + 
                           self.cds.state_units[ self.y_axis ] , 
                           fontsize = self.fontsize)
        
        self.ax.set_xlim([ self.x_axis_min , self.x_axis_max ])
        self.ax.set_ylim([ self.y_axis_min , self.y_axis_max ])
        self.ax.tick_params( labelsize = self.fontsize )
        self.ax.grid(True)
        self.phasefig.tight_layout()
        
        
    ###########################################################################
    def plot(self):
        """ Plot phase plane """
        
        self.compute_grid()
        self.plot_init()
        self.compute_vector_field()
        self.plot_vector_field()
        self.plot_finish()
        self.phasefig.show()
        
        
###############################################################################
# 3D Phase Plot Object for phase plane analysis
###############################################################################
        
        
class PhasePlot3( PhasePlot ):
    """ 
    Continous dynamic system phase plot 3D
    ---------------------------------------------------
    x_axis : index of state to display as x axis
    y_axis : index of state to display as y axis
    z_axis : index of state to display as z axis
    
    """
    ###########################################################################
    def __init__(self, ContinuousDynamicSystem, x_axis=0,  y_axis=1, z_axis=2):
        
        super().__init__(ContinuousDynamicSystem, x_axis, y_axis)
        
        # Smaller resolution
        self.x_axis_n   = 5
        self.y_axis_n   = 5
        self.z_axis_n   = 5
        
        # Z axis
        self.z_axis     = z_axis  
        self.z_axis_min = self.cds.x_lb[ self.z_axis ]
        self.z_axis_max = self.cds.x_ub[ self.z_axis ]
        
        # Plotting params
        self.color      = 'r'
        self.dpi        = 200
        self.linewidth  = 0.5
        self.length     = 0.2
        self.arrowstyle = '->'
        self.fontsize   = 6
        
        
    ###########################################################################
    def compute_grid(self):
        
        x = np.linspace( self.x_axis_min , self.x_axis_max , self.x_axis_n )
        y = np.linspace( self.y_axis_min , self.y_axis_max , self.y_axis_n )
        z = np.linspace( self.z_axis_min , self.z_axis_max , self.z_axis_n )
        
        self.X, self.Y, self.Z = np.meshgrid( x, y, z)
        
            
    ###########################################################################
    def compute_vector_field(self):
        
        self.v = np.zeros(self.X.shape)
        self.w = np.zeros(self.Y.shape)
        self.u = np.zeros(self.Z.shape)
        
        for i in range(self.x_axis_n):
            for j in range(self.y_axis_n):
                for k in range(self.z_axis_n):
                
                    # Actual states
                    x  = np.copy( self.xbar )    # default value for all states
                    x[ self.x_axis ] = self.X[i, j, k]
                    x[ self.y_axis ] = self.Y[i, j, k]
                    x[ self.z_axis ] = self.Z[i, j, k]
                    
                    # States derivative open loop
                    dx = self.f( x , self.ubar , self.t ) 
                    
                    # Assign vector components
                    self.v[i,j,k] = dx[ self.x_axis ]
                    self.w[i,j,k] = dx[ self.y_axis ]
                    self.u[i,j,k] = dx[ self.z_axis ]
                    
                       
    ###########################################################################
    def plot_vector_field(self):
        try:
            self.ax = self.phasefig.axes[0]
        except IndexError:
            self.ax = self.phasefig.add_subplot(111, projection='3d')

        self.ax.quiver( self.X, self.Y, self.Z, self.v, self.w, self.u, 
                       color=self.color,  linewidth = self.linewidth,
                       length = self.length)
                       #, headlength = self.headlength, normalize = True )
        
        
    ###########################################################################
    def plot_finish(self):
        
        self.ax.set_xlabel(self.cds.state_label[ self.x_axis ] + ' ' +
        self.cds.state_units[ self.x_axis ] , fontsize = self.fontsize)
        self.ax.set_ylabel(self.cds.state_label[ self.y_axis ] + ' ' +
        self.cds.state_units[ self.y_axis ] , fontsize = self.fontsize)
        self.ax.set_zlabel(self.cds.state_label[ self.z_axis ] + ' ' +
        self.cds.state_units[ self.z_axis ] , fontsize = self.fontsize)
        
        plt.grid(True)
        plt.tight_layout()
        

