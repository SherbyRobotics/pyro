##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


##############################################################################
# Map Base Class
##############################################################################
class Map:

    ############################
    def __init__(self):
        """ """

        self.obstacles = []

    ############################
    def collision_check(self, x, y):
        collision = False

        return collision

    #############################
    def traversability(self, x, y):
        """Traversability map 1 is good 0 is impossible"""
        trav = 1.0

        return trav

    #############################
    def height(self, x, y):
        """Compute height using from (x,y) position"""
        z = 0.0

        return z

    #############################
    def visualize(self, x_range=(0, 10), y_range=(0, 10), res=0.1, mode="2d"):
        """
        Visualizes the map.

        Args:
            x_range (tuple): (min, max) for x axis.
            y_range (tuple): (min, max) for y axis.
            res (float): Resolution of the grid.
            mode (str): '2d' for contour/heatmap, '3d' for surface plot.
        """

        x = np.arange(x_range[0], x_range[1], res)
        y = np.arange(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        vec_height = np.vectorize(self.height)
        vec_trav = np.vectorize(self.traversability)

        Z = vec_height(X, Y)
        T = vec_trav(X, Y)

        fig = plt.figure(figsize=(10, 8))

        if mode == "3d":
            ax = fig.add_subplot(111, projection="3d")

            # Plot surface: Height is Z, Color is Traversability
            # We normalize Traversability to map it to a colormap
            norm_t = plt.Normalize(0.0, 1.0)
            colors = plt.cm.RdYlGn(norm_t(T))  # Red (low trav) to Green (high trav)

            surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False)
            ax.set_title("3D Terrain: Height (Geometry) & Traversability (Color)")
            ax.set_zlabel("Height")

        else:  # Default 2D
            ax = fig.add_subplot(111)

            # Background: Traversability heatmap
            # cmap='RdYlGn' maps low values to Red, high to Green
            mesh = ax.pcolormesh(
                X, Y, T, shading="auto", cmap="RdYlGn", vmin=0, vmax=1, alpha=0.8
            )
            plt.colorbar(mesh, label="Traversability Index")

            # Overlay: Height Contours
            contours = ax.contour(X, Y, Z, colors="black", alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=8)

            # Overlay: Obstacles
            for ox, oy, r in self.obstacles:
                # Create a circle patch
                # fill=True, hatch='//' implies a solid physical obstacle
                circle = Circle(
                    (ox, oy),
                    r,
                    edgecolor="black",
                    facecolor="gray",
                    alpha=0.9,
                    hatch="//",
                    label="Obstacle",
                )
                ax.add_patch(circle)

            # Handle legend (prevent duplicate labels)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")

            ax.set_title(
                "2D Map: Traversability (Color), Height (Lines), Obstacles (Gray)"
            )
            ax.set_aspect("equal")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        plt.tight_layout()
        plt.show()

        return fig, ax


##############################################################################
# Basic Map exemple
##############################################################################


class GaussianMapWithObstacles(Map):
    """2D Map with circular obstacles, traversability and height defined by Gaussian mixtures."""

    ############################
    def __init__(self):
        """Initialize map parameters."""

        # Obstacles [x, y, radius]
        self.obstacles = [[2, 2, 0.5], [3, 8, 0.2], [7, 2, 0.5], [1, 3, 1.5]]

        # Traversability map parameters
        # Gaussian mixture: [(x, y, amplitude, sigma_x, sigma_y), ...]
        self.traversability_params = [
            (9, 8, 3.5, 1, 3),
            (2, 2, 1.2, 2.5, 1.5),
        ]

        # Height map parameters
        # Gaussian mixture: [(x, y, amplitude, sigma_x, sigma_y), ...]
        self.height_params = [
            (1, 5, 2, 1, 1),
            (3, 2, -0.8, 1.5, 1.5),
            (1, 2, 3, 1, 1),
            (2, 9, 4, 2, 2),
        ]

    ############################
    def collision_check(self, x, y):
        """Check for collision with circular obstacles."""

        collision = False
        for ox, oy, r in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r**2:
                collision = True

        return collision

    #############################
    def traversability(self, x, y, check_collision=False):
        """Compute traversability (0 to 1) using a Gaussian mixture model."""

        if check_collision:
            if self.collision_check(x, y) == True:
                return 0.0

        trav = 0.0
        for gx, gy, amp, sigma_x, sigma_y in self.traversability_params:
            # Summing positive costs
            trav += amp * np.exp(
                -(
                    ((x - gx) ** 2) / (2 * sigma_x**2)
                    + ((y - gy) ** 2) / (2 * sigma_y**2)
                )
            )

        # Traversability is 1.0 by default but decays to 0 as gaussian amplitude get bigger
        return np.exp(-trav)

    #############################
    def height(self, x, y):
        """Compute height using a Gaussian mixture model."""
        z = 0.0
        for gx, gy, amp, sigma_x, sigma_y in self.height_params:
            z += amp * np.exp(
                -(
                    ((x - gx) ** 2) / (2 * sigma_x**2)
                    + ((y - gy) ** 2) / (2 * sigma_y**2)
                )
            )
        return z


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    my_map = Map()
    my_map = GaussianMapWithObstacles()

    # Check simple collision
    print(f"Collision at (5,5): {my_map.collision_check(0, 0)}")

    # Visualize in 2D
    print("Generating 2D Map...")
    my_map.visualize(mode="2d")

    # Visualize in 3D
    print("Generating 3D Map...")
    my_map.visualize(mode="3d")
