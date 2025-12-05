##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

##############################################################################
from pyro.dynamic import system
from pyro.planning.ugv_map import Map, GaussianMapWithObstacles

##############################################################################


##  UGV Base Class with Map and Planner Method


class UGV:
    """abstract UGV base class with map and planner API"""

    ############################
    def __init__(self, map=None):
        """ """

        # Map
        if map is None:
            self.map = Map()
        else:
            self.map = map

    ###########################
    def get_xy_from_state(self, x):
        """Extract (x,y) position from state vector x"""

        return x[0], x[1]

    ###########################
    def plot_map(self, x, y, range=10):
        """Plot the map and the position of the UGV on the graph"""

        if not hasattr(self, "map"):
            raise AttributeError("UGV object has no map attribute.")

        # Plot the map
        fig, ax = self.map.visualize(
            x_range=(x - range, x + range),
            y_range=(y - range, y + range),
            res=0.1,
            mode="2d",
        )

        # Plot the UGV position
        ax.plot(x, y, "ro", label="UGV Position")  # 'ro' for red circle
        ax.legend()
        plt.show()

        return fig, ax

    ############################
    # API for planners

    ###########################
    def distance(self, x1, x2):

        return np.linalg.norm(x1 - x2)

    ###########################
    def steer(self, x0, xf, step_size=0.5):
        ### Return a new state moved from x0 toward xf by step_size ###

        dx = xf - x0
        dist = np.linalg.norm(dx)

        if dist < step_size:
            return xf

        return x0 + (dx / dist) * step_size

    ###########################
    def get_path_points(self, x1, x2, resolution=0.1):
        """Return a discretize path for collision checking"""

        dist = self.distance(x1, x2)
        steps = int(dist / resolution) + 2
        return np.linspace(x1, x2, steps)

    ###########################
    def add_path_on_ax(self, x1, x2, ax):
        """Plot a line between x1 and x2"""

        pts = self.get_path_points(x1, x2)

        ax.plot(pts[:, 0], pts[:, 1], "k-")


##############################################################################


#
class UGV_Particule(system.ContinuousDynamicSystem, UGV):
    """
    Holonomic 2D point-robot
    -----------------------------------
    dx   = u[0]
    dy   = u[1]
    """

    ############################
    def __init__(self, map=None):
        """ """

        # Map
        if map is None:
            self.map = Map()
        else:
            self.map = map

        # Dimensions
        self.n = 2
        self.m = 2
        self.p = 2

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Holonomic Mobile Robot"
        self.state_label = ["x", "y"]
        self.input_label = ["vx", "vy"]
        self.output_label = ["x", "y"]

        # Units
        self.state_units = ["[m]", "[m]"]
        self.input_units = ["[m/sec]", "[m/sec]"]
        self.output_units = ["[m]", "[m]"]

        # State working range
        self.x_ub = np.array([10, 10])
        self.x_lb = np.array([-10, -10])

    #############################
    def f(self, x=np.zeros(3), u=np.zeros(2), t=0):
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

        dx = np.zeros(self.n)  # State derivative vector

        dx[0] = u[0]
        dx[1] = u[1]

        return dx

    ###########################################################################
    # For graphical output
    ###########################################################################

    #############################
    def xut2q(self, x, u, t):
        """compute config q"""

        q = x  # kinematic model : state = config space

        return q

    ###########################################################################
    def forward_kinematic_domain(self, q):
        """ """
        l = 10

        domain = [(-l, l), (-l, l), (-l, l)]  #

        return domain

    ###########################################################################
    def forward_kinematic_lines(self, q):
        """
        Compute points p = [x;y;z] positions given config q
        ----------------------------------------------------
        - points of interest for ploting

        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines

        """

        lines_pts = []  # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        ###########################
        # Top line
        ###########################

        pts = np.zeros((5, 3))

        d = 0.2

        pts[0, 0] = q[0] + d
        pts[0, 1] = q[1] + d
        pts[1, 0] = q[0] + d
        pts[1, 1] = q[1] - d
        pts[2, 0] = q[0] - d
        pts[2, 1] = q[1] - d
        pts[3, 0] = q[0] - d
        pts[3, 1] = q[1] + d
        pts[4, 0] = q[0] + d
        pts[4, 1] = q[1] + d

        lines_pts.append(pts)
        lines_style.append("-")
        lines_color.append("b")

        return lines_pts, lines_style, lines_color

    ###########################
    def isavalidstate(self, x):
        """check if x is in the state domain"""

        out_of_bounds = False

        for i in range(self.n):
            out_of_bounds = out_of_bounds or (x[i] < self.x_lb[i])
            out_of_bounds = out_of_bounds or (x[i] > self.x_ub[i])

        x, y = x[0], x[1]
        collision = self.map.collision_check(x, y)

        return not (out_of_bounds or collision)

    ############
    # API for planners
    ############

    ###########################
    def distance(self, x1, x2):

        return np.linalg.norm(x1 - x2)

    ###########################
    def steer(self, x0, xf, step_size=0.5):
        ### Return a new state moved from x0 toward xf by step_size ###

        dx = xf - x0
        dist = np.linalg.norm(dx)

        if dist < step_size:
            return xf

        return x0 + (dx / dist) * step_size

    ###########################
    def get_path_points(self, x1, x2, resolution=0.1):
        """Return a discretize path for collision checking"""

        dist = self.distance(x1, x2)
        steps = int(dist / resolution) + 2
        return np.linspace(x1, x2, steps)

    ###########################
    def plot_path(
        self,
        x1,
        x2,
        ax=None,
        color="g",
        alpha=0.2,
        linewidth=0.5,
    ):
        """Plot a line between x1 and x2"""

        if ax is None:
            ax = plt.gca()

        ax.plot(
            [x1[0], x2[0]],
            [x1[1], x2[1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        return ax


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    sys = UGV_Particule(map=GaussianMapWithObstacles())

    fig, ax = sys.plot_map(0, 0)
    sys.add_path_on_ax(np.array([0, 0]), np.array([5, 5]), ax)

    # sys.ubar = np.array([1.0, 1.0])
    # sys.x0 = np.array([0.0, 0.0])
    # sys.compute_trajectory(tf=5.0)
    # sys.plot_trajectory(plot="xu")
    # sys.animate_simulation()
