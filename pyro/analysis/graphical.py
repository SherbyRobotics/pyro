# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:05:55 2018

@author: Alexandre
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

from pyro.analysis import phaseanalysis

# Embed font type in PDF
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

###############################################################################

class TrajectoryPlotter:
    def __init__(self, cds):
        self.sys = cds

        # Ploting
        self.fontsize = 5
        self.figsize  = (4, 3)
        self.dpi      = 300

    def plot(self, traj, plot = 'x' , show = True):
        """
        Create a figure with trajectories for states, inputs, outputs and cost
        ----------------------------------------------------------------------
        plot = 'All'
        plot = 'xu'
        plot = 'xy'
        plot = 'x'
        plot = 'u'
        plot = 'y'
        plot = 'j'
        """

        if 'j' in plot and (traj.J is None or traj.dJ is None):
            raise ValueError(
                "Trajectory does not contain cost data but plotting 'j' was requested"
            )

        sys = self.sys

        # For closed-loop systems, extract the inner Dynamic system for plotting
        try:
            sys = self.sys.cds # sys is the global system
        except AttributeError:
            pass

        # Number of subplots
        if plot == 'All':
            l = sys.n + sys.m + sys.p + 2
        elif plot == 'xuj':
            l = sys.n + sys.m + 2
        elif plot == 'xu':
            l = sys.n + sys.m
        elif plot == 'xy':
            l = sys.n + sys.p
        elif plot == 'x':
            l = sys.n
        elif plot == 'u':
            l = sys.m
        elif plot == 'y':
            l = sys.p
        elif plot == 'j':
            l = 2
        else:
            raise ValueError('not a valid ploting argument')

        simfig , plots = plt.subplots(l, sharex=True, figsize=self.figsize,
                                      dpi=self.dpi, frameon=True)

        #######################################################################
        #Fix bug for single variable plotting
        if l == 1:
            plots = [plots]
        #######################################################################

        simfig.canvas.set_window_title('Trajectory for ' + self.sys.name)

        j = 0 # plot index

        if plot=='All' or plot=='x' or plot=='xu' or plot=='xy' or plot=='xuj':
            # For all states
            for i in range( sys.n ):
                plots[j].plot( traj.t , traj.x[:,i] , 'b')
                plots[j].set_ylabel(sys.state_label[i] +'\n'+
                sys.state_units[i] , fontsize=self.fontsize )
                plots[j].grid(True)
                plots[j].tick_params( labelsize = self.fontsize )
                j = j + 1

        if plot == 'All' or plot == 'u' or plot == 'xu' or plot == 'xuj':
            # For all inputs
            for i in range( sys.m ):
                plots[j].plot( traj.t , traj.u[:,i] , 'r')
                plots[j].set_ylabel(sys.input_label[i] + '\n' +
                sys.input_units[i] , fontsize=self.fontsize )
                plots[j].grid(True)
                plots[j].tick_params( labelsize = self.fontsize )
                j = j + 1

        if plot == 'All' or plot == 'y' or plot == 'xy':
            # For all outputs
            for i in range( sys.p ):
                plots[j].plot( traj.t , traj.y[:,i] , 'k')
                plots[j].set_ylabel(sys.output_label[i] + '\n' +
                sys.output_units[i] , fontsize=self.fontsize )
                plots[j].grid(True)
                plots[j].tick_params( labelsize = self.fontsize )
                j = j + 1

        if plot == 'All' or plot == 'j' or plot == 'xuj':
            # Cost function
            plots[j].plot( traj.t , traj.dJ[:] , 'b')
            plots[j].set_ylabel('dJ', fontsize=self.fontsize )
            plots[j].grid(True)
            plots[j].tick_params( labelsize = self.fontsize )
            j = j + 1
            plots[j].plot( traj.t , traj.J[:] , 'r')
            plots[j].set_ylabel('J', fontsize=self.fontsize )
            plots[j].grid(True)
            plots[j].tick_params( labelsize = self.fontsize )
            j = j + 1

        plots[l-1].set_xlabel('Time [sec]', fontsize=self.fontsize )

        simfig.tight_layout()

        if show:
            simfig.show()

        self.fig   = simfig
        self.plots = plots

    def phase_plane_trajectory(self, traj, x_axis=0, y_axis=1):
        """ """
        pp = phaseanalysis.PhasePlot( self.sys , x_axis , y_axis )
        pp.plot()

        plt.plot(traj.x[:,x_axis], traj.x[:,y_axis], 'b-') # path
        plt.plot([traj.x[0,x_axis]], [traj.x[0,y_axis]], 'o') # start
        plt.plot([traj.x[-1,x_axis]], [traj.x[-1,y_axis]], 's') # end

        pp.phasefig.tight_layout()

    ###########################################################################
    def phase_plane_trajectory_3d(self, traj, x_axis=0, y_axis=1, z_axis=2):
        """ """
        pp = phaseanalysis.PhasePlot3( self.sys , x_axis, y_axis, z_axis)

        pp.plot()

        pp.ax.plot(traj.x[:,x_axis],
                        traj.x[:,y_axis],
                        traj.x[:,z_axis],
                        'b-') # path
        pp.ax.plot([traj.x[0,x_axis]],
                        [traj.x[0,y_axis]],
                        [traj.x[0,z_axis]],
                        'o') # start
        pp.ax.plot([traj.x[-1,x_axis]],
                        [traj.x[-1,y_axis]],
                        [traj.x[-1,z_axis]],
                        's') # end

        pp.ax.set_xlim( self.sys.x_lb[ x_axis ] ,
                             self.sys.x_ub[ x_axis ])
        pp.ax.set_ylim( self.sys.x_lb[ y_axis ] ,
                             self.sys.x_ub[ y_axis ])
        pp.ax.set_zlim( self.sys.x_lb[ z_axis ] ,
                             self.sys.x_ub[ z_axis ])

        pp.phasefig.tight_layout()

            ###########################################################################
    def phase_plane_trajectory_closed_loop(self, traj, x_axis, y_axis):
        """ """
        pp = phaseanalysis.PhasePlot( self.sys , x_axis , y_axis )

        pp.compute_grid()
        pp.plot_init()

        # Closed-loop Behavior
        pp.color = 'r'
        pp.compute_vector_field()
        pp.plot_vector_field()

        # Open-Loop Behavior
        pp.f     = self.sys.f
        pp.ubar  = self.sys.ubar
        pp.color = 'b'
        pp.compute_vector_field()
        pp.plot_vector_field()

        pp.plot_finish()

        # Plot trajectory
        plt.plot(traj.x[:,x_axis], traj.x[:,y_axis], 'b-') # path
        plt.plot([traj.x[0,x_axis]], [traj.x[0,y_axis]], 'o') # start
        plt.plot([traj.x[-1,x_axis]], [traj.x[-1,y_axis]], 's') # end

        plt.tight_layout()
        pp.phasefig.show()
        
        
        

class Animator:
    """ 

    """
    
    ###########################################################################
    def __init__(self, sys ):
        """
        
        sys = system.ContinuousDynamicSystem()
        
        sys needs to implement:
        
        get configuration from states, inputs and time
        ----------------------------------------------
        q             = sys.xut2q( x , u , t )
        
        get graphic output list of lines from configuration
        ----------------------------------------------
        lines_pts     = sys.forward_kinematic_lines( q )
        
        get graphic domain from configuration
        ----------------------------------------------
        ((,),(,),(,)) = sys.forward_kinematic_domain( q )
        
        """
        
        self.sys = sys
        
        self.x_axis = 0
        self.y_axis = 1
        
        # Params
        self.figsize   = (4, 3)
        self.dpi       = 300
        self.linestyle = 'o-'
        self.fontsize  = 5

    ###########################################################################
    def show(self, q , x_axis = 0 , y_axis = 1 ):
        """ Plot figure of configuration q """
        
        # Update axis to plot in 2D
        
        self.x_axis = x_axis
        self.y_axis = y_axis
        
        # Get data
        lines_pts      = self.sys.forward_kinematic_lines( q )
        domain         = self.sys.forward_kinematic_domain( q )
        
        # Plot
        self.showfig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.showfig.canvas.set_window_title('2D Configuration of ' + 
                                            self.sys.name )
        self.showax = self.showfig.add_subplot(111, autoscale_on=False )
        self.showax.grid()
        self.showax.axis('equal')
        self.showax.set_xlim(  domain[x_axis] )
        self.showax.set_ylim(  domain[y_axis] )
        
        self.showlines = []
        
        for pts in lines_pts:
            x_pts = pts[:, x_axis ]
            y_pts = pts[:, y_axis ]
            line  = self.showax.plot( x_pts, y_pts, self.linestyle)
            self.showlines.append( line )

        plt.draw()
        plt.show()
        
    
    ###########################################################################
    def show3(self, q ):
        """ Plot figure of configuration q """
        
        # Get data
        lines_pts      = self.sys.forward_kinematic_lines( q )
        domain         = self.sys.forward_kinematic_domain( q )
        
        # Plot
        self.show3fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.show3fig.canvas.set_window_title('3D Configuration of ' + 
                                            self.sys.name )
        self.show3ax = self.show3fig.gca(projection='3d')
                
        self.show3lines = []
        
        for pts in lines_pts:
            x_pts = pts[:, 0 ]
            y_pts = pts[:, 1 ]
            z_pts = pts[:, 2 ]
            line  = self.show3ax.plot( x_pts, y_pts, z_pts, self.linestyle)
            self.show3lines.append( line )
            
        self.show3ax.set_xlim3d( domain[0] )
        self.show3ax.set_xlabel('X')
        self.show3ax.set_ylim3d( domain[1] )
        self.show3ax.set_ylabel('Y')
        self.show3ax.set_zlim3d( domain[2] )
        self.show3ax.set_zlabel('Z')
        
        plt.show()
        
    

    ###########################################################################
    def animate_simulation(self, traj, time_factor_video =  1.0 , is_3d = False, 
                                 save = False , file_name = 'Animation' ):
        """ 
        Show Animation of the simulation 
        ----------------------------------
        time_factor_video < 1 --> Slow motion video        
        
        """  
        self.is_3d = is_3d
        
        # Init list
        self.ani_lines_pts = []
        self.ani_domains   = []

        nsteps = traj.t.size
        self.sim_dt = (traj.t[-1] - traj.t[0]) / (traj.t.size - 1)

        # For all simulation data points
        for i in range( nsteps ):
            # Get configuration q from simulation
            q               = self.sys.xut2q(traj.x[i,:] ,
                                             traj.u[i,:] , 
                                             traj.t[i] )
            
            #TODO fix dependency on sys.sim
            
            # Compute graphical forward kinematic
            lines_pts       = self.sys.forward_kinematic_lines( q )
            domain          = self.sys.forward_kinematic_domain( q )
            # Save data in lists
            self.ani_lines_pts.append(lines_pts)
            self.ani_domains.append(domain)
            
        # Init figure
        self.ani_fig = plt.figure(figsize=self.figsize, dpi=self.dpi )
        
        
        if is_3d:
            self.ani_ax = p3.Axes3D( self.ani_fig )
            self.ani_ax.set_xlim3d(self.ani_domains[0][0])
            self.ani_ax.set_xlabel('X')
            self.ani_ax.set_ylim3d(self.ani_domains[0][1])
            self.ani_ax.set_ylabel('Y')
            self.ani_ax.set_zlim3d(self.ani_domains[0][2])
            self.ani_ax.set_zlabel('Z')
            self.ani_fig.canvas.set_window_title('3D Animation of ' + 
                                            self.sys.name )
        else:
            self.ani_ax = self.ani_fig.add_subplot(111, autoscale_on=True)
            self.ani_ax.axis('equal')
            self.ani_ax.set_xlim(  self.ani_domains[0][self.x_axis] )
            self.ani_ax.set_ylim(  self.ani_domains[0][self.y_axis] )
            self.ani_fig.canvas.set_window_title('2D Animation of ' + 
                                            self.sys.name )
            
        self.ani_ax.tick_params(axis='both', which='both', labelsize=
                                self.fontsize)
        self.ani_ax.grid()
                
        # Plot lines at t=0
        self.lines = []
        
        # for each lines of the t=0 data point
        for line_pts in self.ani_lines_pts[0]:
            if is_3d:
                thisx = line_pts[:,0]
                thisy = line_pts[:,1]
                thisz = line_pts[:,2]
                line, = self.ani_ax.plot(thisx, thisy, thisz, self.linestyle)
                self.time_text = self.ani_ax.text(0, 0, 0, 'time =', 
                                                  transform=
                                                  self.ani_ax.transAxes)
            else:
                thisx = line_pts[:,self.x_axis]
                thisy = line_pts[:,self.y_axis]
                line, = self.ani_ax.plot(thisx, thisy, self.linestyle)
                self.time_text = self.ani_ax.text(0.05, 0.9, 'time =', 
                                                  transform=self.
                                                  ani_ax.transAxes)
                self.ani_fig.tight_layout()
            self.lines.append( line )
        
        self.time_template = 'time = %.1fs'
        
        # Animation
        inter      =  40.             # ms --> 25 frame per second
        frame_dt   =  inter / 1000. 
        
        if ( frame_dt * time_factor_video )  < self.sim_dt :
            # Simulation is slower than video
            
            # don't skip steps
            self.skip_steps = 1
            
            # adjust frame speed to simulation                                    
            inter           = self.sim_dt * 1000. / time_factor_video 
            
            n_frame         = nsteps
            
        else:
            # Simulation is faster than video
            
            # --> number of simulation frame to skip between video frames
            factor          =  frame_dt / self.sim_dt * time_factor_video
            self.skip_steps =  int( factor  ) 
            
            # --> number of video frames
            n_frame         =  int( nsteps / self.skip_steps )
        
        # ANIMATION
        # blit=True option crash on mac
        #self.ani = animation.FuncAnimation( self.ani_fig, self.__animate__, 
        # n_frame , interval = inter , init_func=self.__ani_init__ , blit=True)
        
        if self.is_3d:
            self.ani = animation.FuncAnimation( self.ani_fig, self.__animate__, 
                                                n_frame , interval = inter )
        else:
            self.ani = animation.FuncAnimation( self.ani_fig, self.__animate__,
                                                n_frame , interval = inter, 
                                                init_func=self.__ani_init__ )
        if save:
            self.ani.save( file_name + '.html' ) # , writer = 'mencoder' )

        self.ani_fig.show()
        
    #####################################    
    def __ani_init__(self):
        for line in self.lines:
            line.set_data([], [])
        self.time_text.set_text('')
        return self.lines, self.time_text, self.ani_ax
    
    ######################################
    def __animate__(self,i):
        # Update lines
        for j, line in enumerate(self.lines):
            if self.is_3d:
                thisx = self.ani_lines_pts[i * self.skip_steps][j][:,0]
                thisy = self.ani_lines_pts[i * self.skip_steps][j][:,1]
                thisz = self.ani_lines_pts[i * self.skip_steps][j][:,2]
                line.set_data(thisx, thisy)
                line.set_3d_properties(thisz)
            else:
                thisx = self.ani_lines_pts[i*self.skip_steps][j][:,self.x_axis]
                thisy = self.ani_lines_pts[i*self.skip_steps][j][:,self.y_axis]
                line.set_data(thisx, thisy)
            
        # Update time
        self.time_text.set_text(self.time_template % 
                                ( i * self.skip_steps * self.sim_dt )
                                )
        
        # Update domain
        if self.is_3d:
            self.ani_ax.set_xlim3d( self.ani_domains[i * self.skip_steps][0] )
            self.ani_ax.set_ylim3d( self.ani_domains[i * self.skip_steps][1] )
            self.ani_ax.set_zlim3d( self.ani_domains[i * self.skip_steps][2] )
        else:
            i_x = self.x_axis
            i_y = self.y_axis
            self.ani_ax.set_xlim( self.ani_domains[i * self.skip_steps][i_x] )
            self.ani_ax.set_ylim( self.ani_domains[i * self.skip_steps][i_y] )
        
        return self.lines, self.time_text, self.ani_ax
    

'''
###############################################################################
##################          Main                         ######################
###############################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic import pendulum
    from pyro.dynamic import vehicle
    
    #sys  = SinglePendulum()
    #x0   = np.array([0,1])
    
    #sys.plot_trajectory( x0 )
    
    #sys.show( np.array([0]))
    #sys.show3( np.array([0]))
    
    #sys.animate_simulation()
    
    sys = pendulum.DoublePendulum()
    x0 = np.array([0.1,0.1,0,0])
    
    #sys.show(np.array([0.1,0.1]))
    #sys.show3(np.array([0.1,0.1]))
    
    is_3d = False
    
    sys.plot_trajectory( x0 , 20)
    
    a = Animator(sys)
    a.animate_simulation(1,is_3d)
    
    sys = vehicle.KinematicBicyleModel()
    sys.ubar = np.array([1,0.01])
    x0 = np.array([0,0,0])
    
    b = Animator(sys)
    sys.plot_trajectory( x0 , 100 )
    b.animate_simulation(10,is_3d)