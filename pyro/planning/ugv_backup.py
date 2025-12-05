import numpy as np
import matplotlib.pyplot as plt
import copy
import math


# ---------------------------------------------------------
# 1. Dubins Path Logic (Geometric Calculation)
# ---------------------------------------------------------
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

    def plot(
        self, start, end, ax=None, show_circles=False, show_arrows=False, **kwargs
    ):
        """
        Visualizes the Dubins path.
        Options:
            show_circles: Plot the turning circles for the start and end segments.
            show_arrows: Plot orientation arrows along the path.
        """
        cost, mode, segments = self.calculate(start, end)
        points = self.interpolate(start, cost, mode, segments)
        points = np.array(points)

        if ax is None:
            ax = plt.gca()

        # Plot main line
        ax.plot(points[:, 0], points[:, 1], **kwargs)

        # Plot Arrows
        if show_arrows and len(points) > 1:
            # Subsample points for arrows to avoid clutter
            step = max(1, len(points) // 10)
            for i in range(0, len(points), step):
                x, y, yaw = points[i]
                ax.arrow(
                    x,
                    y,
                    0.4 * np.cos(yaw),
                    0.4 * np.sin(yaw),
                    head_width=0.15,
                    color=kwargs.get("color", "b"),
                    alpha=0.8,
                )

        # Plot Construction Circles
        if show_circles and mode is not None:
            self._plot_turning_circles(start, end, mode, ax)

    def _plot_turning_circles(self, start, end, mode, ax):
        """Helper to draw the start and end turning circles based on mode."""
        sx, sy, syaw = start
        ex, ey, eyaw = end
        r = self.radius

        # Function to get circle center given pose and 'L' or 'R'
        def get_center(x, y, yaw, turn_type):
            if turn_type == "L":
                cx = x - r * np.sin(yaw)
                cy = y + r * np.cos(yaw)
            elif turn_type == "R":
                cx = x + r * np.sin(yaw)
                cy = y - r * np.cos(yaw)
            else:
                return None
            return cx, cy

        # Start Circle
        start_center = get_center(sx, sy, syaw, mode[0])
        if start_center:
            c1 = plt.Circle(
                start_center, r, color="k", fill=False, linestyle="--", alpha=0.4
            )
            ax.add_patch(c1)
            # Mark center
            ax.plot(start_center[0], start_center[1], "k+", alpha=0.4)

        # End Circle
        end_center = get_center(ex, ey, eyaw, mode[2])
        if end_center:
            c2 = plt.Circle(
                end_center, r, color="k", fill=False, linestyle="--", alpha=0.4
            )
            ax.add_patch(c2)
            # Mark center
            ax.plot(end_center[0], end_center[1], "k+", alpha=0.4)

        # Middle Circle (for RLR / LRL cases)
        # Note: Calculation of the middle circle center for RLR/LRL is complex
        # as it depends on the intermediate point. Skipping for visual simplicity
        # unless strictly needed, but Start/End are the most informative.

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


# ---------------------------------------------------------
# 2. Dynamic Systems (Pyro API Compatible)
# ---------------------------------------------------------


class DynamicSystem:
    """Base class mocking pyro.dynamic.system.ContinuousDynamicSystem"""

    def __init__(self, n, m):
        self.n = n  # State dimension
        self.m = m  # Control dimension
        self.x_lb = None  # State lower bounds
        self.x_ub = None  # State upper bounds
        self.u_lb = None  # Control lower bounds
        self.u_ub = None  # Control upper bounds
        self.name = "Generic System"

    def f(self, x, u, t=0):
        """Dynamics: dx/dt = f(x, u, t)"""
        raise NotImplementedError

    def is_free(self, x):
        """Check if state is valid (free space)"""
        raise NotImplementedError

    def plot_path(self, x1, x2, ax=None, **kwargs):
        """Visualizes the path between two states."""
        raise NotImplementedError


class Holonomic2D(DynamicSystem):
    def __init__(self):
        super().__init__(n=2, m=2)
        self.name = "Holonomic Point Mass"
        self.x_lb = np.array([0, 0])
        self.x_ub = np.array([10, 10])
        self.u_lb = np.array([-1, -1])  # Max velocity
        self.u_ub = np.array([1, 1])

        self.obstacles = [[7, 5, 3.5], [3, 8, 1.0], [7, 2, 1.0], [2, 3, 1.5]]

    def f(self, x, u, t=0):
        # dx/dt = u (velocity control)
        return u

    def is_free(self, x):
        if np.any(x < self.x_lb) or np.any(x > self.x_ub):
            return False
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(x[0:2] - np.array([ox, oy])) <= r:
                return False
        return True

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def steer(self, x_from, x_to, step_size):
        diff = x_to - x_from
        dist = np.linalg.norm(diff)
        if dist < step_size:
            return x_to
        return x_from + (diff / dist) * step_size

    def get_path_points(self, x1, x2, resolution=0.1):
        dist = self.dist(x1, x2)
        steps = int(dist / resolution) + 2
        return np.linspace(x1, x2, steps)

    def plot_path(self, x1, x2, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        # Filter out Dubins-specific kwargs if any exist to prevent MPL errors
        kwargs.pop("show_circles", None)
        kwargs.pop("show_arrows", None)
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], **kwargs)


class CarSystem(DynamicSystem):
    def __init__(self):
        super().__init__(n=3, m=2)  # x, y, theta; u = [v, phi]
        self.name = "Kinematic Car (Dubins)"
        self.radius = 2.0
        self.dubins = DubinsPath(self.radius)

        self.x_lb = np.array([0, 0, -np.pi])
        self.x_ub = np.array([10, 10, np.pi])
        self.u_lb = np.array([-1, -0.5])  # Min velocity, max right turn
        self.u_ub = np.array([1, 0.5])  # Max velocity, max left turn

        self.obstacles = [[7, 5, 3.5], [3, 8, 1.0], [7, 2, 1.0], [2, 3, 1.5]]

    def f(self, x, u, t=0):
        # Kinematic bicycle model
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = v * tan(phi) / L (simplified here to just turning rate)
        v = u[0]
        w = u[1]  # angular velocity
        return np.array([v * np.cos(x[2]), v * np.sin(x[2]), w])

    def is_free(self, x):
        # Boundary check (ignore theta for bounds)
        if (
            x[0] < self.x_lb[0]
            or x[0] > self.x_ub[0]
            or x[1] < self.x_lb[1]
            or x[1] > self.x_ub[1]
        ):
            return False
        # Obstacles
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(x[0:2] - np.array([ox, oy])) <= r:
                return False
        return True

    def dist(self, x1, x2):
        cost, _, _ = self.dubins.calculate(x1, x2)
        return cost

    def steer(self, x_from, x_to, step_size):
        cost, mode, segments = self.dubins.calculate(x_from, x_to)
        if cost < step_size:
            return x_to
        points = self.dubins.interpolate(
            x_from, cost, mode, segments, step_dist=step_size
        )
        return points[-1]

    def get_path_points(self, x1, x2):
        cost, mode, segments = self.dubins.calculate(x1, x2)
        return self.dubins.interpolate(x1, cost, mode, segments)

    def plot_path(self, x1, x2, ax=None, **kwargs):
        self.dubins.plot(x1, x2, ax, **kwargs)


# ---------------------------------------------------------
# 3. Planners (Pyro API Compatible)
# ---------------------------------------------------------


class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.cost = 0.0


class RRT:
    """
    Standard Rapidly-exploring Random Tree (RRT) Planner
    """

    def __init__(self, sys, x_start, x_goal):
        self.sys = sys
        self.start = Node(x_start)
        self.goal = Node(x_goal)

        # Hyperparameters
        self.max_iter = 500
        self.step_size = 1.0
        self.goal_sample_rate = 0.1
        self.goal_tolerance = 0.5

        self.nodes = [self.start]

    def collision_free_path(self, x1, x2):
        path_points = self.sys.get_path_points(x1, x2)
        for p in path_points:
            if not self.sys.is_free(p):
                return False
        return True

    def get_nearest_node(self, point):
        dists = [self.sys.dist(node.state, point) for node in self.nodes]
        min_index = np.argmin(dists)
        return self.nodes[min_index]

    def plan(self):
        print(f"Starting RRT Planning with {self.sys.name}...")
        for i in range(self.max_iter):
            # 1. Sample
            if np.random.random() < self.goal_sample_rate:
                x_rand = self.goal.state
            else:
                x_rand = np.random.uniform(self.sys.x_lb, self.sys.x_ub)

            # 2. Nearest
            nearest_node = self.get_nearest_node(x_rand)

            # 3. Steer
            x_new_state = self.sys.steer(nearest_node.state, x_rand, self.step_size)

            # 4. Check & Add
            if self.sys.is_free(x_new_state) and self.collision_free_path(
                nearest_node.state, x_new_state
            ):
                new_node = Node(x_new_state)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.sys.dist(
                    nearest_node.state, x_new_state
                )
                self.nodes.append(new_node)

        return self.extract_path()

    def extract_path(self):
        # Find node closest to goal
        best_dist = float("inf")
        goal_node = None
        for node in self.nodes:
            d = self.sys.dist(node.state, self.goal.state)
            if d < self.goal_tolerance and d < best_dist:
                best_dist = d
                goal_node = node

        if goal_node is None:
            print("Goal not reached! Showing closest path.")
            dists = [self.sys.dist(n.state, self.goal.state) for n in self.nodes]
            goal_node = self.nodes[np.argmin(dists)]

        path = []
        curr = goal_node
        while curr is not None:
            if curr.parent:
                segment = self.sys.get_path_points(curr.parent.state, curr.state)
                path.append(segment[::-1])
            else:
                path.append([curr.state])
            curr = curr.parent
        return np.vstack(path[::-1])

    def plot(self, path=None):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        # Obstacles
        for ox, oy, r in self.sys.obstacles:
            circle = plt.Circle((ox, oy), r, color="r", alpha=0.5)
            ax.add_patch(circle)

        # Tree
        for node in self.nodes:
            if node.parent:
                self.sys.plot_path(
                    node.parent.state,
                    node.state,
                    ax=ax,
                    color="g",
                    alpha=0.2,
                    linewidth=0.5,
                )

        # Path (Redrawn using system plot_path for generic support)
        if path is not None and len(path) > 0:
            # We must iterate through the extracted path nodes if we want to use Dubins logic
            # However, extract_path returns a dense array.
            # Simple fallback: Plot dense array for generic line
            plt.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="Path")

            # Draw Start/Goal
            plt.plot(
                self.start.state[0],
                self.start.state[1],
                "bs",
                markersize=10,
                label="Start",
            )
            plt.plot(
                self.goal.state[0],
                self.goal.state[1],
                "k*",
                markersize=10,
                label="Goal",
            )

            # Start/Goal Arrows (if applicable)
            if self.sys.n >= 3:
                s = self.start.state
                g = self.goal.state
                ax.arrow(
                    s[0],
                    s[1],
                    0.6 * np.cos(s[2]),
                    0.6 * np.sin(s[2]),
                    head_width=0.2,
                    color="b",
                )
                ax.arrow(
                    g[0],
                    g[1],
                    0.6 * np.cos(g[2]),
                    0.6 * np.sin(g[2]),
                    head_width=0.2,
                    color="k",
                )

        plt.xlim(self.sys.x_lb[0], self.sys.x_ub[0])
        plt.ylim(self.sys.x_lb[1], self.sys.x_ub[1])
        plt.legend()
        plt.title(f"{type(self).__name__} Planning ({self.sys.name})")
        plt.grid(True)
        plt.show()


class RRTStar(RRT):
    """
    RRT* Planner (Inherits from RRT, adds rewiring)
    """

    def __init__(self, sys, x_start, x_goal):
        super().__init__(sys, x_start, x_goal)
        self.search_radius = 2.0

    def get_neighbors(self, new_state):
        neighbors = []
        for node in self.nodes:
            if self.sys.dist(node.state, new_state) <= self.search_radius:
                neighbors.append(node)
        return neighbors

    def plan(self):
        print(f"Starting RRT* Planning with {self.sys.name}...")
        for i in range(self.max_iter):
            # 1. Sample
            if np.random.random() < self.goal_sample_rate:
                x_rand = self.goal.state
            else:
                x_rand = np.random.uniform(self.sys.x_lb, self.sys.x_ub)

            # 2. Nearest
            nearest_node = self.get_nearest_node(x_rand)

            # 3. Steer
            x_new_state = self.sys.steer(nearest_node.state, x_rand, self.step_size)

            # 4. Check
            if self.sys.is_free(x_new_state) and self.collision_free_path(
                nearest_node.state, x_new_state
            ):

                new_node = Node(x_new_state)

                # 5. Choose Best Parent (RRT* Magic)
                neighbors = self.get_neighbors(x_new_state)
                min_cost = nearest_node.cost + self.sys.dist(
                    nearest_node.state, x_new_state
                )
                best_parent = nearest_node

                for neighbor in neighbors:
                    cost_via_neighbor = neighbor.cost + self.sys.dist(
                        neighbor.state, x_new_state
                    )
                    if cost_via_neighbor < min_cost:
                        if self.collision_free_path(neighbor.state, x_new_state):
                            min_cost = cost_via_neighbor
                            best_parent = neighbor

                new_node.parent = best_parent
                new_node.cost = min_cost
                self.nodes.append(new_node)

                # 6. Rewire
                for neighbor in neighbors:
                    cost_via_new_node = new_node.cost + self.sys.dist(
                        new_node.state, neighbor.state
                    )
                    if cost_via_new_node < neighbor.cost:
                        if self.collision_free_path(new_node.state, neighbor.state):
                            neighbor.parent = new_node
                            neighbor.cost = cost_via_new_node

        return self.extract_path()


# ---------------------------------------------------------
# Execution
