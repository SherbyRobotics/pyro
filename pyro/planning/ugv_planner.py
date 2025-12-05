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
                new_node = Node(x_new_state)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.sys.distance(
                    nearest_node.state, x_new_state
                )
                self.nodes.append(new_node)

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


if __name__ == "__main__":

    from pyro.planning.ugv_map import GaussianMapWithObstacles

    from pyro.planning.ugv_model import UGV_Particule

    sys = UGV_Particule(map=GaussianMapWithObstacles())

    # fig, ax = sys.plot_map(0, 0)
    # sys.add_path_on_ax(np.array([0, 0]), np.array([5, 5]), ax)

    planner = RRT(sys)
    planner.set_start_goal([0, 0], [8, 8])
    planner.max_iter = 2000
    planner.step_size = 0.1
    path = planner.compute_solution()
    # planner.plot_tree()
    planner.show_solution(show_tree=True)
