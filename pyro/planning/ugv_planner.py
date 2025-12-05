##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pyro.planning.plan import Planner


class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.cost = 0.0


###############################################################################
class RRT(Planner):
    """ """

    ############################
    def __init__(self, sys, cost_function=None):
        """ """

        # Dynamic system model and constraints
        self.sys = sys

        # Cost function
        if cost_function is None:
            self.cost_function = sys.cost_function  # default is quadratic cost
        else:
            self.cost_function = cost_function

        # Start and goal state
        self.set_start_goal(sys.x0, sys.xbar)

        # Output variable
        self.traj = None

        # Tree initialization
        self.start = Node(self.x_start)
        self.nodes = [self.start]

        # RRT Hyperparameters
        self.max_iter = 500
        self.step_size = 0.5
        self.goal_sample_rate = 0.1
        self.goal_tolerance = 1.0

    #############################
    def set_start_goal(self, x_start, x_goal):
        self.x_start = np.array(x_start)
        self.x_goal = np.array(x_goal)

        # Init tree
        self.start = Node(self.x_start)
        self.nodes = [self.start]

    ##############################
    def show_solution(self, ax=None, show_tree=True):
        """Plot computed trajectory solution"""

        if not hasattr(self, "traj_x"):
            print("No solution computed yet. Run compute_solution() first.")
            return

        if ax is None:
            x_range = (self.sys.x_lb[0], self.sys.x_ub[0])
            y_range = (self.sys.x_lb[1], self.sys.x_ub[1])
            fig, ax = self.sys.map.visualize(
                mode="2d", x_range=x_range, y_range=y_range
            )

        # Option to overlay the search tree
        if show_tree:
            self.plot_tree(ax=ax, alpha=0.3)

        # Plot the final path
        # traj_x is a numpy array of shape (N, 2)
        ax.plot(
            self.traj_x[:, 0],
            self.traj_x[:, 1],
            "r-",
            linewidth=3,
            label="Solution Path",
        )

        # Plot Start and Goal
        ax.plot(self.x_start[0], self.x_start[1], "bs", markersize=10, label="Start")
        ax.plot(self.x_goal[0], self.x_goal[1], "k*", markersize=10, label="Goal")

        ax.legend(loc="lower left")
        plt.show()

    ##############################
    def collision_free_path(self, x1, x2):

        path_points = self.sys.get_path_points(x1, x2)
        for x in path_points:
            if not self.sys.isavalidstate(x):
                return False
        return True

    ##############################
    def get_nearest_node(self, point):

        dists = [self.sys.distance(node.state, point) for node in self.nodes]
        min_index = np.argmin(dists)
        return self.nodes[min_index]

    ##############################
    def _add_node(self, state, parent):
        """Helper to add node, can be overridden by subclasses"""
        new_node = Node(state)
        new_node.parent = parent
        new_node.cost = parent.cost + self.sys.distance(parent.state, state)
        self.nodes.append(new_node)
        return new_node

    ##############################
    def compute_solution(self):

        print(f"Starting RRT Planning with {self.sys.name}...")

        for i in range(self.max_iter):
            # 1. Sample
            if np.random.random() < self.goal_sample_rate:
                x_rand = self.x_goal
            else:
                x_rand = np.random.uniform(self.sys.x_lb, self.sys.x_ub)

            # 2. Nearest
            nearest_node = self.get_nearest_node(x_rand)

            # 3. Steer
            x_new_state = self.sys.steer(nearest_node.state, x_rand, self.step_size)

            # 4. Check & Add
            if self.collision_free_path(nearest_node.state, x_new_state):
                self._add_node(x_new_state, nearest_node)

        self.solution_path = self.extract_path()

        return self.solution_path

    ##############################
    def extract_path(self):

        # Find node closest to goal
        best_dist = float("inf")
        goal_node = None
        for node in self.nodes:
            d = self.sys.distance(node.state, self.x_goal)
            if d < self.goal_tolerance and d < best_dist:
                best_dist = d
                goal_node = node

        if goal_node is None:
            print("Goal not reached! Choosing the closest path.")
            dists = [self.sys.distance(n.state, self.x_goal) for n in self.nodes]
            goal_node = self.nodes[np.argmin(dists)]

        self.goal_node = goal_node

        # Backtrack from goal to start
        path_x = []
        path_nodes = []
        curr = goal_node
        while curr is not None:
            path_x.append(curr.state)
            path_nodes.append(curr)
            curr = curr.parent

        self.traj_x = np.array(path_x[::-1])
        self.traj_nodes = path_nodes[::-1]

        return self.traj_x

    ##############################
    def plot_tree_on_map(self, ax=None, alpha=0.3):
        self.sys.plot_map(ax=ax)
        self.plot_tree(ax=ax, alpha=alpha)

    ##############################
    def plot_tree(self, ax=None, alpha=1.0):

        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()

        # Tree
        for node in self.nodes:
            if node.parent:
                self.sys.plot_path(
                    node.parent.state,
                    node.state,
                    ax=ax,
                    color="g",
                    alpha=alpha,
                    linewidth=0.5,
                )

        # Start and Goal
        ax.plot(
            self.x_start[0],
            self.x_start[1],
            "bo",
            markersize=8,
            label="Start",
        )

        ax.add_patch(
            Circle(
                self.x_goal,
                self.goal_tolerance,
                color="r",
                alpha=0.2,
                label="Goal Region",
            )
        )

        if ax is None:
            plt.xlim(self.sys.x_lb[0], self.sys.x_ub[0])
            plt.ylim(self.sys.x_lb[1], self.sys.x_ub[1])
            plt.legend()
            plt.title(f"{type(self).__name__} Planning ({self.sys.name})")
            plt.grid(True)
            plt.show()


##############################################################################
###############################################################################
class RRTStar(RRT):
    """
    RRT* Planner (Inherits from RRT, adds rewiring)
    """

    ############################
    def __init__(self, sys, cost_function=None):
        super().__init__(sys, cost_function)

        self.search_radius = 2.0  # Radius to look for neighbors

    ##############################
    def get_neighbors(self, new_state):

        neighbors = []
        for node in self.nodes:
            if self.sys.distance(node.state, new_state) <= self.search_radius:
                neighbors.append(node)
        return neighbors

    ##############################
    def _add_node(self, state, nearest_node):
        """
        Override _add_node to include RRT* logic:
        1. Choose best parent from neighbors
        2. Add node
        3. Rewire neighbors
        """

        # Find neighbors within search radius
        neighbors = self.get_neighbors(state)

        # Default parent is nearest_node
        min_cost = nearest_node.cost + self.sys.distance(nearest_node.state, state)
        best_parent = nearest_node

        # 1. Choose Best Parent (optimization step)
        for neighbor in neighbors:
            # Calculate potential cost
            cost_via_neighbor = neighbor.cost + self.sys.distance(neighbor.state, state)

            # Check if this neighbor offers a cheaper path
            if cost_via_neighbor < min_cost:
                if self.collision_free_path(neighbor.state, state):
                    min_cost = cost_via_neighbor
                    best_parent = neighbor

        # Create new node with best parent found
        new_node = Node(state)
        new_node.parent = best_parent
        new_node.cost = min_cost
        self.nodes.append(new_node)

        # 2. Rewire Neighbors (rewiring step)
        for neighbor in neighbors:
            # Calculate cost if we route through new_node
            cost_via_new_node = new_node.cost + self.sys.distance(
                new_node.state, neighbor.state
            )

            # If cheaper, rewire
            if cost_via_new_node < neighbor.cost:
                if self.collision_free_path(new_node.state, neighbor.state):
                    neighbor.parent = new_node
                    neighbor.cost = cost_via_new_node
                    # Note: We are not recursively updating costs of children here for simplicity,
                    # but typically RRT* would propagate this cost improvement down the branch.

        return new_node


if __name__ == "__main__":

    from pyro.planning.ugv_map import GaussianMapWithObstacles

    from pyro.planning.ugv_model import UGV_Particule

    sys = UGV_Particule(map=GaussianMapWithObstacles())

    # fig, ax = sys.plot_map(0, 0)
    # sys.add_path_on_ax(np.array([0, 0]), np.array([5, 5]), ax)

    planner = RRT(sys)
    # planner = RRTStar(sys)
    planner.set_start_goal([0, 0], [8, 8])
    planner.max_iter = 2000
    planner.step_size = 0.1
    path = planner.compute_solution()
    # planner.plot_tree()
    planner.show_solution(show_tree=True)

    planner = RRTStar(sys)
    planner.set_start_goal([0, 0], [8, 8])
    planner.max_iter = 2000
    planner.step_size = 0.1
    path = planner.compute_solution()
    # planner.plot_tree()
    planner.show_solution(show_tree=True)
