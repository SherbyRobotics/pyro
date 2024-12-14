# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:05:55 2018

@author: Alexandre
"""

import sys as python_system
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from pyro.analysis import phaseanalysis


###############################################################################
#  Note: Modify here matplolib setting to fit your environment
###############################################################################

# Use interactive backend
try:
    # Default usage for interactive mode
    matplotlib.use("Qt5Agg")
    plt.ion()  # Set interactive mode

except:

    try:
        # For MacOSX
        matplotlib.use("MacOSX")
        plt.ion()

    except:

        try:
            # For MacOSX
            matplotlib.use("macosx")
            plt.ion()

        except:

            print("Warning: Could not load validated backend mode for matplotlib")
            print(
                "Matplotlib list of interactive backends:",
                matplotlib.backends.backend_registry.list_builtin(
                    matplotlib.backends.BackendFilter.INTERACTIVE
                ),
            )
            plt.ion()  # Set interactive mode


# Default figure settings
default_figsize = (4, 3)
default_dpi = 250
default_linestyle = "-"
default_fontsize = 5

# True if running in IPython, False if running the file in terminal
if hasattr(python_system, "ps1"):
    figure_blocking = False  # Set to not block the code when showing a figure
else:
    # We want to block figure to prevent the script from terminating
    figure_blocking = True  # Set to block the code when showing a figure

# Embed font type in PDF when exporting
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


print("Matplotlib backend:", plt.get_backend())
print("Matplotlib interactive:", matplotlib.is_interactive())
# print('Matplotlib list of interactive backends:', matplotlib.rcsetup.interactive_bk)
print("Matplotlib figure blocking:", figure_blocking)


###############################################################################
class TrajectoryPlotter:

    ##########################################################################
    def __init__(self, sys):

        self.sys = sys

        # Ploting
        self.fontsize = default_fontsize
        self.figsize = default_figsize
        self.dpi = default_dpi

    ##########################################################################
    def plot(self, traj, plot="x", show=True):
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
        plot = 'z'
        """

        if "j" in plot and (traj.J is None or traj.dJ is None):
            raise ValueError(
                "Trajectory does not contain cost data but plotting 'j' was requested"
            )

        sys = self.sys

        # For closed-loop systems, extract the inner Dynamic system for plotting
        # try:
        #    sys = self.sys.plant # sys is the global system
        # except AttributeError:
        #    pass

        # Number of subplots
        if plot == "All":
            l = sys.n + sys.m + sys.p + 2
        elif plot == "xuj":
            l = sys.n + sys.m + 2
        elif plot == "xu":
            l = sys.n + sys.m
        elif plot == "xy":
            l = sys.n + sys.p
        elif plot == "x":
            l = sys.n
        elif plot == "u":
            l = sys.m
        elif plot == "y":
            l = sys.p
        elif plot == "j":
            l = 2
        elif plot == "z":
            l = sys.controller.l
        else:
            raise ValueError("not a valid ploting argument")

        simfig, plots = plt.subplots(
            l, sharex=True, figsize=self.figsize, dpi=self.dpi, frameon=True
        )

        lines = [None] * l

        #######################################################################
        # Fix bug for single variable plotting
        if l == 1:
            plots = [plots]
        #######################################################################

        simfig.canvas.manager.set_window_title("Trajectory for " + self.sys.name)

        j = 0  # plot index

        if (
            plot == "All"
            or plot == "x"
            or plot == "xu"
            or plot == "xy"
            or plot == "xuj"
        ):
            # For all states
            for i in range(sys.n):
                lines[j] = plots[j].plot(traj.t, traj.x[:, i], "b")[0]
                plots[j].set_ylabel(
                    sys.state_label[i] + "\n" + sys.state_units[i],
                    fontsize=self.fontsize,
                )
                plots[j].grid(True)
                plots[j].tick_params(labelsize=self.fontsize)
                j = j + 1

        if plot == "All" or plot == "u" or plot == "xu" or plot == "xuj":
            # For all inputs
            for i in range(sys.m):
                lines[j] = plots[j].plot(traj.t, traj.u[:, i], "r")[0]
                plots[j].set_ylabel(
                    sys.input_label[i] + "\n" + sys.input_units[i],
                    fontsize=self.fontsize,
                )
                plots[j].grid(True)
                plots[j].tick_params(labelsize=self.fontsize)
                j = j + 1

        if plot == "All" or plot == "y" or plot == "xy":
            # For all outputs
            for i in range(sys.p):
                lines[j] = plots[j].plot(traj.t, traj.y[:, i], "k")[0]
                plots[j].set_ylabel(
                    sys.output_label[i] + "\n" + sys.output_units[i],
                    fontsize=self.fontsize,
                )
                plots[j].grid(True)
                plots[j].tick_params(labelsize=self.fontsize)
                j = j + 1

        if plot == "All" or plot == "j" or plot == "xuj":
            # Cost function
            lines[j] = plots[j].plot(traj.t, traj.dJ[:], "b")[0]
            plots[j].set_ylabel("dJ", fontsize=self.fontsize)
            plots[j].grid(True)
            plots[j].tick_params(labelsize=self.fontsize)
            j = j + 1
            lines[j] = plots[j].plot(traj.t, traj.J[:], "r")[0]
            plots[j].set_ylabel("J", fontsize=self.fontsize)
            plots[j].grid(True)
            plots[j].tick_params(labelsize=self.fontsize)
            j = j + 1

        if plot == "z":
            # Internal states
            n = sys.n - sys.controller.l
            for i in range(l):
                lines[j] = plots[j].plot(traj.t, traj.x[:, i + n], "b")[0]
                plots[j].set_ylabel(
                    sys.state_label[i + n] + "\n" + sys.state_units[i + n],
                    fontsize=self.fontsize,
                )
                plots[j].grid(True)
                plots[j].tick_params(labelsize=self.fontsize)
                j = j + 1

        plots[l - 1].set_xlabel("Time [sec]", fontsize=self.fontsize)

        simfig.tight_layout()

        simfig.canvas.draw()
        plt.draw()
        plt.show(block=figure_blocking)

        self.fig = simfig
        self.plots = plots
        self.lines = lines
        self.l = l

        return (simfig, plots, lines)

    ##########################################################################
    def update_plot(self, traj, plot="x"):
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
        plot = 'z'
        """

        lines = self.lines
        sys = self.sys
        plots = self.plots

        j = 0

        if (
            plot == "All"
            or plot == "x"
            or plot == "xu"
            or plot == "xy"
            or plot == "xuj"
        ):

            # For all states
            for i in range(sys.n):
                lines[j].set_data(traj.t, traj.x[:, i])
                plots[j].relim()
                plots[j].autoscale_view(True, True, True)
                j = j + 1

        if plot == "All" or plot == "u" or plot == "xu" or plot == "xuj":
            # For all inputs
            for i in range(sys.m):
                lines[j].set_data(traj.t, traj.u[:, i])
                plots[j].relim()
                plots[j].autoscale_view(True, True, True)
                j = j + 1

        if plot == "All" or plot == "y" or plot == "xy":
            # For all outputs
            for i in range(sys.p):
                lines[j].set_data(traj.t, traj.y[:, i])
                plots[j].relim()
                plots[j].autoscale_view(True, True, True)
                j = j + 1

        if plot == "All" or plot == "j" or plot == "xuj":
            # Cost function
            lines[j].set_data(traj.t, traj.dJ[:, i])
            plots[j].relim()
            plots[j].autoscale_view(True, True, True)
            j = j + 1
            lines[j].set_data(traj.t, traj.J[:, i])
            plots[j].relim()
            plots[j].autoscale_view(True, True, True)
            j = j + 1

        if plot == "z":
            # Internal states
            n = sys.n - sys.controller.l
            for i in range(self.l):
                lines[j].set_data(traj.t, traj.x[:, i + n])
                plots[j].relim()
                plots[j].autoscale_view(True, True, True)
                j = j + 1

    ##########################################################################
    def phase_plane_trajectory(self, traj, x_axis=0, y_axis=1):
        """ """
        pp = phaseanalysis.PhasePlot(self.sys, x_axis, y_axis)
        pp.plot()

        plt.plot(traj.x[:, x_axis], traj.x[:, y_axis], "b-")  # path
        plt.plot([traj.x[0, x_axis]], [traj.x[0, y_axis]], "ko")  # start
        plt.plot([traj.x[-1, x_axis]], [traj.x[-1, y_axis]], "rx")  # end

        plt.draw()

        pp.phasefig.tight_layout()

        plt.draw()
        plt.show(block=figure_blocking)

    ###########################################################################
    def phase_plane_trajectory_3d(self, traj, x_axis=0, y_axis=1, z_axis=2):
        """ """
        pp = phaseanalysis.PhasePlot3(self.sys, x_axis, y_axis, z_axis)

        pp.plot()

        pp.ax.plot(
            traj.x[:, x_axis], traj.x[:, y_axis], traj.x[:, z_axis], "b-"
        )  # path
        pp.ax.plot(
            [traj.x[0, x_axis]], [traj.x[0, y_axis]], [traj.x[0, z_axis]], "o"
        )  # start
        pp.ax.plot(
            [traj.x[-1, x_axis]], [traj.x[-1, y_axis]], [traj.x[-1, z_axis]], "s"
        )  # start # end

        pp.ax.set_xlim(self.sys.x_lb[x_axis], self.sys.x_ub[x_axis])
        pp.ax.set_ylim(self.sys.x_lb[y_axis], self.sys.x_ub[y_axis])
        pp.ax.set_zlim(self.sys.x_lb[z_axis], self.sys.x_ub[z_axis])

        plt.draw()

        pp.phasefig.tight_layout()

        plt.draw()
        plt.show(block=figure_blocking)

    ###########################################################################
    def phase_plane_trajectory_closed_loop(self, traj, x_axis, y_axis):
        """ """
        pp = phaseanalysis.PhasePlot(self.sys, x_axis, y_axis)

        pp.compute_grid()
        pp.plot_init()

        # Closed-loop Behavior
        pp.color = "r"
        pp.compute_vector_field()
        pp.plot_vector_field()

        # Open-Loop Behavior
        pp.f = self.sys.f
        pp.ubar = self.sys.ubar
        pp.color = "b"
        pp.compute_vector_field()
        pp.plot_vector_field()

        pp.plot_finish()

        # Plot trajectory
        plt.plot(traj.x[:, x_axis], traj.x[:, y_axis], "b-")  # path
        plt.plot([traj.x[0, x_axis]], [traj.x[0, y_axis]], "o")  # start
        plt.plot([traj.x[-1, x_axis]], [traj.x[-1, y_axis]], "s")  # end

        plt.draw()

        plt.tight_layout()

        plt.draw()
        plt.show(block=figure_blocking)


##########################################################################
# Animator
##########################################################################
class Animator:
    """ """

    ###########################################################################
    def __init__(self, sys):
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

        # Ploting Param
        self.fontsize = default_fontsize
        self.figsize = default_figsize
        self.dpi = default_dpi
        self.linestyle = sys.linestyle

        # Label
        self.top_right_label = None

    ###########################################################################
    def show(self, q, x_axis=0, y_axis=1):
        """Plot figure of configuration q"""

        # Update axis to plot in 2D

        self.x_axis = x_axis
        self.y_axis = y_axis

        # Get data
        lines = self.sys.forward_kinematic_lines(q)
        domain = self.sys.forward_kinematic_domain(q)

        # Two type of output are supported for foward_kinematic_lines
        # check if return is a tuple of ( pts , style , color )
        # or only the pts (then use default values)

        if type(lines) is tuple:
            # If the foward_kinematic_lines function specify style and color
            lines_pts = lines[0]
            lines_style = lines[1]
            lines_color = lines[2]
        else:
            lines_pts = lines
            lines_style = []
            lines_color = []
            for j, line in enumerate(lines):
                lines_style.append(self.sys.linestyle)  # default value
                lines_color.append(self.sys.linescolor)  # default value

        # Plot
        self.showfig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.showfig.canvas.manager.set_window_title(
            "2D Configuration of " + self.sys.name
        )
        self.showax = self.showfig.add_subplot(111, autoscale_on=False)
        self.showax.grid()
        self.showax.axis("equal")
        self.showax.set_xlim(domain[x_axis])
        self.showax.set_ylim(domain[y_axis])
        self.showax.tick_params(axis="both", which="both", labelsize=self.fontsize)

        self.showlines = []

        for j, pts in enumerate(lines_pts):

            x_pts = pts[:, x_axis]
            y_pts = pts[:, y_axis]
            linestyle = lines_style[j] + lines_color[j]
            (line,) = self.showax.plot(x_pts, y_pts, linestyle)

            self.showlines.append(line)

        plt.show(block=figure_blocking)

    ###########################################################################
    def show3(self, q):
        """Plot figure of configuration q"""

        # Get data
        lines = self.sys.forward_kinematic_lines(q)
        domain = self.sys.forward_kinematic_domain(q)

        if type(lines) is tuple:
            # If the foward_kinematic_lines function specify style and color
            lines_pts = lines[0]
            lines_style = lines[1]
            lines_color = lines[2]
        else:
            lines_pts = lines
            lines_style = []
            lines_color = []
            for j, line in enumerate(lines):
                lines_style.append(self.sys.linestyle)  # default value
                lines_color.append(self.sys.linescolor)  # default value

        # Plot
        self.show3fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.show3fig.canvas.manager.set_window_title(
            "3D Configuration of " + self.sys.name
        )
        self.show3ax = self.show3fig.add_subplot(projection="3d")

        self.show3lines = []

        for j, pts in enumerate(lines_pts):
            x_pts = pts[:, 0]
            y_pts = pts[:, 1]
            z_pts = pts[:, 2]
            linestyle = lines_style[j] + lines_color[j]
            (line,) = self.show3ax.plot(x_pts, y_pts, z_pts, linestyle)

            self.show3lines.append(line)

        self.show3ax.set_xlim3d(domain[0])
        self.show3ax.set_xlabel("X")
        self.show3ax.set_ylim3d(domain[1])
        self.show3ax.set_ylabel("Y")
        self.show3ax.set_zlim3d(domain[2])
        self.show3ax.set_zlabel("Z")
        self.show3ax.tick_params(axis="both", which="both", labelsize=self.fontsize)

        plt.show(block=figure_blocking)

    ###########################################################################
    def show_plus(self, x, u, t, x_axis=0, y_axis=1):
        """Plot figure of system at state x"""

        # Update axis to plot in 2D

        self.x_axis = x_axis
        self.y_axis = y_axis

        # Get data
        lines_data = self.get_lines(x, u, t)

        # Save data in lists for the whole trajectory
        lines_pts = lines_data[0]
        lines_style = lines_data[1]
        lines_color = lines_data[2]
        lines_plus_pts = lines_data[3]
        lines_plus_style = lines_data[4]
        lines_plus_color = lines_data[5]
        domain = lines_data[6]

        # Plot
        self.showfig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.showfig.canvas.manager.set_window_title("2D plot of " + self.sys.name)
        self.showax = self.showfig.add_subplot(111, autoscale_on=False)
        self.showax.grid()
        self.showax.axis("equal")
        self.showax.set_xlim(domain[x_axis])
        self.showax.set_ylim(domain[y_axis])
        self.showax.tick_params(axis="both", which="both", labelsize=self.fontsize)

        self.showlines = []
        self.showlines_plus = []

        # for each lines
        for j, line_pts in enumerate(lines_pts):

            linestyle = lines_style[j] + lines_color[j]

            thisx = line_pts[:, self.x_axis]
            thisy = line_pts[:, self.y_axis]
            (line,) = self.showax.plot(thisx, thisy, linestyle)

            self.showfig.tight_layout()

            self.showlines.append(line)

        # Lines plus optionnal
        if self.sys.lines_plus:

            for j, line_pts in enumerate(lines_plus_pts):

                linestyle = lines_plus_style[j] + lines_plus_color[j]

                thisx = line_pts[:, self.x_axis]
                thisy = line_pts[:, self.y_axis]
                (line,) = self.showax.plot(thisx, thisy, linestyle)

                self.showlines_plus.append(line)

        plt.show(block=figure_blocking)

    ###########################################################################
    def show_plus_update(self, x, u, t):
        """Update a show plus figure"""

        # Get data
        lines_data = self.get_lines(x, u, t)

        # Line data
        lines_pts = lines_data[0]
        lines_style = lines_data[1]
        lines_color = lines_data[2]
        lines_plus_pts = lines_data[3]
        lines_plus_style = lines_data[4]
        lines_plus_color = lines_data[5]
        domain = lines_data[6]

        # Update lines
        for j, line in enumerate(self.showlines):
            thisx = lines_pts[j][:, self.x_axis]
            thisy = lines_pts[j][:, self.y_axis]
            line.set_data(thisx, thisy)

        if self.sys.lines_plus:
            # Update lines plus
            for j, line in enumerate(self.showlines_plus):
                thisx = lines_plus_pts[j][:, self.x_axis]
                thisy = lines_plus_pts[j][:, self.y_axis]
                line.set_data(thisx, thisy)

        # Update domain
        i_x = self.x_axis
        i_y = self.y_axis
        self.showax.set_xlim(domain[i_x])
        self.showax.set_ylim(domain[i_y])

        self.showfig.canvas.draw()

        plt.show(block=figure_blocking)

    ###########################################################################
    def get_lines(self, x, u, t):
        """
        shorcut to get all graphic output data
        """

        # Get configuration q from simulation
        q = self.sys.xut2q(x, u, t)

        # Compute graphical forward kinematic
        lines = self.sys.forward_kinematic_lines(q)
        lines_plus = self.sys.forward_kinematic_lines_plus(x, u, t)
        domain = self.sys.forward_kinematic_domain(q)

        # Two type of output are supported for foward_kinematic_lines
        # check if return is a tuple of ( pts , style , color )
        # or only the pts (then use default values)

        if type(lines) is tuple:
            # If the foward_kinematic_lines function specify style and color
            lines_pts = lines[0]
            lines_style = lines[1]
            lines_color = lines[2]
        else:
            lines_pts = lines
            lines_style = []
            lines_color = []
            for j, line in enumerate(lines):
                lines_style.append(self.sys.linestyle)  # default value
                lines_color.append(self.sys.linescolor)  # default value

        if type(lines_plus) is tuple:
            # If the foward_kinematic_lines function specify style and color
            lines_plus_pts = lines_plus[0]
            lines_plus_style = lines_plus[1]
            lines_plus_color = lines_plus[2]
        else:
            lines_plus_pts = lines_plus
            lines_plus_style = []
            lines_plus_color = []
            for j, line_plus in enumerate(lines_plus):
                lines_plus_style.append(self.sys.linestyle_plus)  # default value
                lines_plus_color.append(self.sys.linescolor_plus)  # default value

        lines_data = (
            lines_pts,
            lines_style,
            lines_color,
            lines_plus_pts,
            lines_plus_style,
            lines_plus_color,
            domain,
        )

        return lines_data

    ###########################################################################
    def animate_simulation(
        self,
        traj,
        time_factor_video=1.0,
        is_3d=False,
        save=False,
        file_name="Animation",
        show=True,
    ):
        """
        Show Animation of the simulation
        ----------------------------------
        time_factor_video < 1 --> Slow motion video

        """
        self.is_3d = is_3d

        # Init lists
        self.ani_lines_pts = []
        self.ani_lines_style = []
        self.ani_lines_color = []

        self.ani_lines_plus_pts = []
        self.ani_lines_plus_style = []
        self.ani_lines_plus_color = []

        self.ani_domains = []

        nsteps = traj.t.size
        self.sim_dt = (traj.t[-1] - traj.t[0]) / (traj.t.size - 1)

        # For all simulation data points
        for i in range(nsteps):

            # Get states , actions , time
            x = traj.x[i, :]
            u = traj.u[i, :]
            t = traj.t[i]

            lines_data = self.get_lines(x, u, t)

            # Save data in lists for the whole trajectory
            self.ani_lines_pts.append(lines_data[0])
            self.ani_lines_style.append(lines_data[1])
            self.ani_lines_color.append(lines_data[2])
            self.ani_lines_plus_pts.append(lines_data[3])
            self.ani_lines_plus_style.append(lines_data[4])
            self.ani_lines_plus_color.append(lines_data[5])
            self.ani_domains.append(lines_data[6])

        # Init figure
        self.ani_fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        if is_3d:
            # self.ani_ax = p3.Axes3D(self.ani_fig) #TODO
            # self.ani_fig.add_axes(self.ani_ax)
            self.ani_ax = self.ani_fig.add_subplot(projection="3d")
            self.ani_ax.set_xlim3d(self.ani_domains[0][0])
            self.ani_ax.set_xlabel("X")
            self.ani_ax.set_ylim3d(self.ani_domains[0][1])
            self.ani_ax.set_ylabel("Y")
            self.ani_ax.set_zlim3d(self.ani_domains[0][2])
            self.ani_ax.set_zlabel("Z")
            self.ani_fig.canvas.manager.set_window_title(
                "3D Animation of " + self.sys.name
            )
        else:
            self.ani_ax = self.ani_fig.add_subplot(111, autoscale_on=True)
            self.ani_ax.axis("equal")
            self.ani_ax.set_xlim(self.ani_domains[0][self.x_axis])
            self.ani_ax.set_ylim(self.ani_domains[0][self.y_axis])
            self.ani_fig.canvas.manager.set_window_title(
                "2D Animation of " + self.sys.name
            )

        self.ani_ax.tick_params(axis="both", which="both", labelsize=self.fontsize)
        self.ani_ax.grid()

        # Plot lines at t=0
        self.lines = []
        self.lines_plus = []

        # for each lines of the t=0 data point
        for j, line_pts in enumerate(self.ani_lines_pts[0]):

            linestyle = self.ani_lines_style[0][j] + self.ani_lines_color[0][j]

            if is_3d:
                thisx = line_pts[:, 0]
                thisy = line_pts[:, 1]
                thisz = line_pts[:, 2]

                (line,) = self.ani_ax.plot(thisx, thisy, thisz, linestyle)
                self.time_text = self.ani_ax.text(
                    0, 0, 0, "time =", transform=self.ani_ax.transAxes
                )
                self.label_text = self.ani_ax.text(0.9, 0.9, 0.9, self.top_right_label)

            else:
                thisx = line_pts[:, self.x_axis]
                thisy = line_pts[:, self.y_axis]
                (line,) = self.ani_ax.plot(thisx, thisy, linestyle)
                self.time_text = self.ani_ax.text(
                    0.05, 0.9, "time =", transform=self.ani_ax.transAxes
                )
                self.label_text = self.ani_ax.text(
                    0.75, 0.8, self.top_right_label, transform=self.ani_ax.transAxes
                )
                self.ani_fig.tight_layout()

            self.lines.append(line)

        # Lines plus optionnal
        if self.sys.lines_plus:

            for j, line_pts in enumerate(self.ani_lines_plus_pts[0]):

                linestyle = (
                    self.ani_lines_plus_style[0][j] + self.ani_lines_plus_color[0][j]
                )

                if is_3d:
                    thisx = line_pts[:, 0]
                    thisy = line_pts[:, 1]
                    thisz = line_pts[:, 2]
                    (line,) = self.ani_ax.plot(thisx, thisy, thisz, linestyle)

                else:
                    thisx = line_pts[:, self.x_axis]
                    thisy = line_pts[:, self.y_axis]
                    (line,) = self.ani_ax.plot(thisx, thisy, linestyle)

                self.lines_plus.append(line)

        self.time_template = "time = %.1fs"

        # Animation
        inter = 40.0  # ms --> 25 frame per second
        frame_dt = inter / 1000.0

        if (frame_dt * time_factor_video) < self.sim_dt:
            # Simulation is slower than video

            # don't skip steps
            self.skip_steps = 1

            # adjust frame speed to simulation
            inter = self.sim_dt * 1000.0 / time_factor_video

            n_frame = nsteps

        else:
            # Simulation is faster than video

            # --> number of simulation frame to skip between video frames
            factor = frame_dt / self.sim_dt * time_factor_video
            self.skip_steps = int(factor)

            # --> number of video frames
            n_frame = int(nsteps / self.skip_steps)

        # ANIMATION
        # blit=True option crash on mac
        # self.ani = animation.FuncAnimation( self.ani_fig, self.__animate__,
        # n_frame , interval = inter , init_func=self.__ani_init__ , blit=True)

        if self.is_3d:
            self.ani = animation.FuncAnimation(
                self.ani_fig, self.__animate__, n_frame, interval=inter
            )
        else:
            self.ani = animation.FuncAnimation(
                self.ani_fig,
                self.__animate__,
                n_frame,
                interval=inter,
                init_func=self.__ani_init__,
            )
        if save:
            self.ani.save(file_name + ".gif", writer="imagemagick", fps=30)

        # self.ani_fig.show( block = figure_blocking )
        if show:
            # plt.ioff()
            plt.show(block=figure_blocking)

        else:
            plt.close(self.ani_fig)

        return self.ani

    #####################################
    def __ani_init__(self):

        for line in self.lines:
            line.set_data([], [])

        for line in self.lines_plus:
            line.set_data([], [])

        self.time_text.set_text("")

        return self.lines, self.lines_plus, self.time_text, self.ani_ax

    ######################################
    def __animate__(self, i):

        # Update lines
        for j, line in enumerate(self.lines):
            if self.is_3d:
                thisx = self.ani_lines_pts[i * self.skip_steps][j][:, 0]
                thisy = self.ani_lines_pts[i * self.skip_steps][j][:, 1]
                thisz = self.ani_lines_pts[i * self.skip_steps][j][:, 2]
                line.set_data(thisx, thisy)
                line.set_3d_properties(thisz)
            else:
                thisx = self.ani_lines_pts[i * self.skip_steps][j][:, self.x_axis]
                thisy = self.ani_lines_pts[i * self.skip_steps][j][:, self.y_axis]
                line.set_data(thisx, thisy)

        if self.sys.lines_plus:
            # Update lines plus
            for j, line in enumerate(self.lines_plus):
                if self.is_3d:
                    thisx = self.ani_lines_plus_pts[i * self.skip_steps][j][:, 0]
                    thisy = self.ani_lines_plus_pts[i * self.skip_steps][j][:, 1]
                    thisz = self.ani_lines_plus_pts[i * self.skip_steps][j][:, 2]
                    line.set_data(thisx, thisy)
                    line.set_3d_properties(thisz)
                else:
                    thisx = self.ani_lines_plus_pts[i * self.skip_steps][j][
                        :, self.x_axis
                    ]
                    thisy = self.ani_lines_plus_pts[i * self.skip_steps][j][
                        :, self.y_axis
                    ]
                    line.set_data(thisx, thisy)

        # Update time
        self.time_text.set_text(
            self.time_template % (i * self.skip_steps * self.sim_dt)
        )

        # Update domain
        if self.is_3d:
            self.ani_ax.set_xlim3d(self.ani_domains[i * self.skip_steps][0])
            self.ani_ax.set_ylim3d(self.ani_domains[i * self.skip_steps][1])
            self.ani_ax.set_zlim3d(self.ani_domains[i * self.skip_steps][2])
        else:
            i_x = self.x_axis
            i_y = self.y_axis
            self.ani_ax.set_xlim(self.ani_domains[i * self.skip_steps][i_x])
            self.ani_ax.set_ylim(self.ani_domains[i * self.skip_steps][i_y])

        return self.lines, self.lines_plus, self.time_text, self.ani_ax


"""
###############################################################################
##################          Main                         ######################
###############################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    from pyro.dynamic import pendulum
    from pyro.dynamic import vehicle_steering

    sys = pendulum.DoublePendulum()
    sys.x0 = np.array([0.1, 0.1, 0, 0])

    sys.plot_phase_plane(0, 2)

    traj = sys.compute_trajectory(2.0)

    plotter = TrajectoryPlotter(sys)
    plotter.plot(traj, "xu")

    sys.x0 = np.array([0.2, 0.1, 0, 0])
    traj2 = sys.compute_trajectory(2.0)

    plotter.update_plot(traj2)

    plotter.phase_plane_trajectory(traj2, 0, 2)

    is_3d = False

    sys.plot_trajectory()

    a = Animator(sys)
    a.animate_simulation(sys.traj, 1, is_3d)

    sys = vehicle_steering.KinematicBicyleModel()
    sys.ubar = np.array([1, 0.01])
    sys.x0 = np.array([0, 0, 0])

    b = Animator(sys)
    b.top_right_label = "this is a label"
    sys.compute_trajectory(100)
    sys.plot_trajectory()
    b.animate_simulation(sys.traj, 10, is_3d)
