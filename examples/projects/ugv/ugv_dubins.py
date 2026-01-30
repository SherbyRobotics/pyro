##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pyro.dynamic import system
from pyro.planning.ugv_model import UGV
from pyro.planning.ugv_map import Map, GaussianMapWithObstacles
from pyro.planning.plan import Planner

##############################################################################
# Dubins Path Logic
##############################################################################


class DubinsPath:
    """
    Handles the calculation of the shortest path between two
    oriented points (x, y, theta) given a turning radius.
    """

    def __init__(self, turning_radius=1.0):
        self.radius = turning_radius

    def _mod2pi(self, theta):
        return theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi)

    def calculate(self, start, end):
        """
        Calculates the shortest Dubins path.
        Returns (cost, mode, segments)
        """
        sx, sy, syaw = start
        ex, ey, eyaw = end
        c = self.radius

        dx = ex - sx
        dy = ey - sy
        D = np.sqrt(dx**2 + dy**2)
        d = D / c  # Normalized distance

        theta = self._mod2pi(np.arctan2(dy, dx))
        alpha = self._mod2pi(syaw - theta)
        beta = self._mod2pi(eyaw - theta)

        # 6 Dubins words: LSL, RSR, LSR, RSL, RLR, LRL
        planners = [self._LSL, self._RSR, self._LSR, self._RSL, self._RLR, self._LRL]
        best_cost = float("inf")
        best_mode = None
        best_t, best_p, best_q = 0, 0, 0

        for planner in planners:
            t, p, q, mode = planner(alpha, beta, d)
            if t is None:
                continue

            # Cost = length * radius
            cost = (abs(t) + abs(p) + abs(q)) * c
            if cost < best_cost:
                best_cost = cost
                best_mode = mode
                best_t, best_p, best_q = t, p, q

        return best_cost, best_mode, (best_t, best_p, best_q)

    def interpolate(self, start, length, mode, segments, step_dist=None):
        """Generates points along the path defined by start, mode, and segments."""
        sx, sy, syaw = start
        t, p, q = segments
        c = self.radius

        # Total lengths of 3 segments
        L1, L2, L3 = t * c, p * c, q * c
        total_len = L1 + L2 + L3

        # If we just want the end point at a specific distance (steer)
        if step_dist is not None:
            distances = [step_dist]
        else:
            # Full path sampling
            distances = np.arange(0, total_len, 0.1)  # 0.1 resolution

            # Handle edge case where path length is 0 or very small (empty array)
            if len(distances) == 0:
                distances = np.array([total_len])
            # Check if last point is far enough from end to avoid duplication
            elif distances[-1] < total_len - 1e-6:
                distances = np.append(distances, total_len)

        points = []
        for dist_covered in distances:
            x, y, yaw = sx, sy, syaw

            # Segment 1
            l_seg = dist_covered
            if l_seg > L1:
                l_seg = L1

            d_phi = (l_seg / c) if mode[0] == "L" else -(l_seg / c)
            x += c * np.sin(yaw + d_phi) - c * np.sin(yaw)
            y += -c * np.cos(yaw + d_phi) + c * np.cos(yaw)
            yaw += d_phi

            dist_remaining = dist_covered - L1

            # Segment 2
            if dist_remaining > 0:
                l_seg = dist_remaining
                if l_seg > L2:
                    l_seg = L2

                if mode[1] == "S":
                    x += l_seg * np.cos(yaw)
                    y += l_seg * np.sin(yaw)
                else:  # Curve
                    d_phi = (l_seg / c) if mode[1] == "L" else -(l_seg / c)
                    x += c * np.sin(yaw + d_phi) - c * np.sin(yaw)
                    y += -c * np.cos(yaw + d_phi) + c * np.cos(yaw)
                    yaw += d_phi

                dist_remaining -= L2

                # Segment 3
                if dist_remaining > 0:
                    l_seg = dist_remaining
                    d_phi = (l_seg / c) if mode[2] == "L" else -(l_seg / c)
                    x += c * np.sin(yaw + d_phi) - c * np.sin(yaw)
                    y += -c * np.cos(yaw + d_phi) + c * np.cos(yaw)
                    yaw += d_phi

            points.append(np.array([x, y, yaw]))

        return points

    # --- Dubins Primitives ---
    def _LSL(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        tmp0 = d + sa - sb
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
        if p_squared < 0:
            return None, None, None, ["L", "S", "L"]

        tmp1 = np.arctan2((cb - ca), tmp0)
        t = self._mod2pi(-alpha + tmp1)
        p = np.sqrt(p_squared)
        q = self._mod2pi(beta - tmp1)
        return t, p, q, ["L", "S", "L"]

    def _RSR(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        tmp0 = d - sa + sb
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
        if p_squared < 0:
            return None, None, None, ["R", "S", "R"]

        tmp1 = np.arctan2((ca - cb), tmp0)
        t = self._mod2pi(alpha - tmp1)
        p = np.sqrt(p_squared)
        q = self._mod2pi(-beta + tmp1)
        return t, p, q, ["R", "S", "R"]

    def _LSR(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
        if p_squared < 0:
            return None, None, None, ["L", "S", "R"]

        p = np.sqrt(p_squared)
        tmp2 = np.arctan2((-ca - cb), (d + sa + sb)) - np.arctan2(-2.0, p)
        t = self._mod2pi(-alpha + tmp2)
        q = self._mod2pi(-self._mod2pi(beta) + tmp2)
        return t, p, q, ["L", "S", "R"]

    def _RSL(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
        if p_squared < 0:
            return None, None, None, ["R", "S", "L"]

        p = np.sqrt(p_squared)
        tmp2 = np.arctan2((ca + cb), (d - sa - sb)) - np.arctan2(2.0, p)
        t = self._mod2pi(alpha - tmp2)
        q = self._mod2pi(beta - tmp2)
        return t, p, q, ["R", "S", "L"]

    def _RLR(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
        if abs(tmp_rlr) > 1.0:
            return None, None, None, ["R", "L", "R"]

        p = self._mod2pi(2 * np.pi - np.arccos(tmp_rlr))
        t = self._mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + p / 2.0)
        q = self._mod2pi(alpha - beta - t + p)
        return t, p, q, ["R", "L", "R"]

    def _LRL(self, alpha, beta, d):
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)

        tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0
        if abs(tmp_lrl) > 1.0:
            return None, None, None, ["L", "R", "L"]

        p = self._mod2pi(2 * np.pi - np.arccos(tmp_lrl))
        t = self._mod2pi(-alpha - np.arctan2(ca - cb, d + sa - sb) + p / 2.0)
        q = self._mod2pi(self._mod2pi(beta) - alpha - t + p)
        return t, p, q, ["L", "R", "L"]


##############################################################################
# Dubins UGV System
##############################################################################


class UGV_Dubins(system.ContinuousDynamicSystem, UGV):
    """
    Non-holonomic robot based on Dubins path (Kinematic Bicycle).
    State: [x, y, theta]
    Inputs: [v, w] (linear velocity, angular velocity)
    """

    ############################
    def __init__(self, map=None, min_turn_radius=1.0):
        """ """

        # Map
        if map is None:
            self.map = Map()
        else:
            self.map = map

        # Dimensions
        self.n = 3  # x, y, theta
        self.m = 2  # v, w
        self.p = 3  # x, y, theta

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Dubins Car"
        self.state_label = ["x", "y", "theta"]
        self.input_label = ["v", "w"]
        self.output_label = ["x", "y", "theta"]

        # Units
        self.state_units = ["[m]", "[m]", "[rad]"]
        self.input_units = ["[m/s]", "[rad/s]"]
        self.output_units = ["[m]", "[m]", "[rad]"]

        # State working range
        self.x_ub = np.array([10, 10, np.pi])
        self.x_lb = np.array([-10, -10, -np.pi])

        # Dubins Helper
        self.radius = min_turn_radius
        self.dubins = DubinsPath(self.radius)

    #############################
    def f(self, x=np.zeros(3), u=np.zeros(2), t=0):
        """
        Continuous time forward dynamics evaluation
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = w
        """

        dx = np.zeros(self.n)
        theta = x[2]
        v = u[0]
        w = u[1]

        dx[0] = v * np.cos(theta)
        dx[1] = v * np.sin(theta)
        dx[2] = w

        return dx

    ###########################
    def isavalidstate(self, x):
        """check if x is in the state domain"""

        # Check Boundaries (ignoring theta wrapping for now)
        if (
            x[0] < self.x_lb[0]
            or x[0] > self.x_ub[0]
            or x[1] < self.x_lb[1]
            or x[1] > self.x_ub[1]
        ):
            return False

        # Check Collision
        collision = self.map.collision_check(x[0], x[1])

        return not collision

    ############
    # API for planners
    ############

    ###########################
    def distance(self, x1, x2):
        """Returns the length of the shortest Dubins path between x1 and x2."""
        cost, _, _ = self.dubins.calculate(x1, x2)
        return cost

    ###########################
    def steer(self, x0, xf, step_size=0.5):
        """
        Return a new state moved from x0 toward xf by step_size
        along the optimal Dubins curve.
        """
        cost, mode, segments = self.dubins.calculate(x0, xf)

        if cost < step_size:
            return xf

        # Get point at exactly step_size distance along the curve
        points = self.dubins.interpolate(x0, cost, mode, segments, step_dist=step_size)
        return points[-1]

    ###########################
    def get_path_points(self, x1, x2, resolution=0.1):
        """Return discretized points along the Dubins path for collision checking."""
        cost, mode, segments = self.dubins.calculate(x1, x2)
        return np.array(self.dubins.interpolate(x1, cost, mode, segments))

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
        """Plot the Dubins curve between x1 and x2"""

        if ax is None:
            ax = plt.gca()

        pts = self.get_path_points(x1, x2)

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        return ax


##############################################################################
# Direct Dubins Planner (Ignores Obstacles)
##############################################################################


class DirectDubinsPlanner(Planner):
    """
    A simple planner that computes the direct Dubins curve
    ignoring obstacles. Useful for testing the system or local steering.
    """

    def __init__(self, sys):
        self.sys = sys
        self.x_start = sys.x0
        self.x_goal = sys.xbar
        self.traj_x = None

    def compute_solution(self):
        print(f"Planning direct Dubins path...")

        # Calculate full path points
        points = self.sys.get_path_points(self.x_start, self.x_goal)
        self.traj_x = np.array(points)

        # Store start/goal explicitly for plotting
        self.traj = self.traj_x  # Alias
        return self.traj_x

    def show_solution(self, ax=None, show_tree=False):
        """Visualize the direct path solution"""

        if self.traj_x is None:
            print("No solution computed yet.")
            return

        if ax is None:
            # Setup plot
            x_range = (self.sys.x_lb[0], self.sys.x_ub[0])
            y_range = (self.sys.x_lb[1], self.sys.x_ub[1])
            fig, ax = self.sys.map.visualize(
                mode="2d", x_range=x_range, y_range=y_range
            )

        # Plot Path
        ax.plot(
            self.traj_x[:, 0], self.traj_x[:, 1], "b-", linewidth=2, label="Dubins Path"
        )

        # Start/Goal Markers
        ax.plot(self.x_start[0], self.x_start[1], "bs", markersize=10, label="Start")
        ax.plot(self.x_goal[0], self.x_goal[1], "k*", markersize=10, label="Goal")

        # Orientation Arrows
        s, g = self.x_start, self.x_goal
        ax.arrow(
            s[0],
            s[1],
            0.6 * np.cos(s[2]),
            0.6 * np.sin(s[2]),
            head_width=0.3,
            color="b",
        )
        ax.arrow(
            g[0],
            g[1],
            0.6 * np.cos(g[2]),
            0.6 * np.sin(g[2]),
            head_width=0.3,
            color="k",
        )

        ax.legend()
        plt.show()


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    from pyro.planning.ugv_planner import RRTStar

    # Create Map
    map = GaussianMapWithObstacles()

    # Create Dubins UGV
    sys = UGV_Dubins(map=map, min_turn_radius=2.0)

    # Set Start and Goal (x, y, theta)
    # Note: Start pointing East (0), Goal pointing North (pi/2)
    sys.x0 = np.array([1.0, 1.0, 0.0])
    sys.xbar = np.array([8.0, 8.0, np.pi / 2])

    # 1. Run Direct Planner
    print("\n--- Direct Dubins Planner ---")
    direct_planner = DirectDubinsPlanner(sys)
    direct_path = direct_planner.compute_solution()
    direct_planner.show_solution()

    # 2. Run RRT* Planner
    print("\n--- RRT* Planner ---")
    rrt_planner = RRTStar(sys)
    rrt_planner.set_start_goal(sys.x0, sys.xbar)
    rrt_planner.max_iter = 1000
    rrt_planner.step_size = 0.8

    rrt_path = rrt_planner.compute_solution()
    rrt_planner.show_solution(show_tree=True)
