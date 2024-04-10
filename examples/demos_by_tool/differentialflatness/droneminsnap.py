# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt

from pyro.analysis import graphical
from pyro.planning import trajectorygeneration
from pyro.dynamic import drone
from pyro.planning import plan

###############################################################################
# Define the c.g. trajectory
###############################################################################

# initial position
x0 = -10.0
y0 = 0.0

# final position
xf = 20.0
yf = 1.0

# time of flight
tf = 4.0

traj_x = trajectorygeneration.SingleAxisPolynomialTrajectoryGenerator(
    poly_N=9, tf=tf, x0=np.array([x0, 0, 0]), xf=np.array([xf, 0, 0])
)

traj_x.Ws[4] = 1.0  # set the weight of the snap term to 1

px, X, t = traj_x.solve()  # find the min snap trajectory

x = X[0, :]
dx = X[1, :]
ax = X[2, :]
dax = X[3, :]
ddax = X[4, :]

traj_y = trajectorygeneration.SingleAxisPolynomialTrajectoryGenerator(
    poly_N=9, tf=tf, x0=np.array([y0, 0, 0]), xf=np.array([yf, 0, 0])
)

traj_y.Ws[4] = 1.0  # set the weight of the snap term to 1

py, Y, t = traj_y.solve()  # find the min snap trajectory

y = Y[0, :]
dy = Y[1, :]
ay = Y[2, :] + 9.81  # added gravity
day = Y[3, :]
dday = Y[4, :]

###############################################################################
# Compute angular trajectory
###############################################################################

theta = np.arctan2(ax, ay)
# theta = np.arctan( (ay/ax))
s = np.sin(theta)
c = np.cos(theta)
dtheta = (dax * c - day * s) / (ax * s + ay * c)
ddtheta = (
    s * (-dday + ay * dtheta**2 - 2 * dax * dtheta)
    + c * (ddax - ax * dtheta**2 - 2 * day * dtheta)
) / (ay * c + ax * s)

###############################################################################
# Compute Forces
###############################################################################

sys = drone.Drone2D()

m = sys.mass
J = sys.inertia
r = sys.truster_offset

F_tot = m * ax / np.sin(theta)
F_del = J * ddtheta / r

T1 = 0.5 * (F_tot + F_del)
T2 = 0.5 * (F_tot - F_del)

###############################################################################
# Plots
###############################################################################

# Create traj
steps = len(t)
us = np.zeros((steps, 2))  # control inputs
xs = np.zeros((steps, 6))
ys = np.zeros((steps, 6))
dxs = np.zeros((steps, 6))

xs[:, 0] = x
xs[:, 1] = y
xs[:, 2] = -theta
xs[:, 3] = dx
xs[:, 4] = dy
xs[:, 5] = -dtheta

us[:, 0] = T1
us[:, 1] = T2

traj = plan.Trajectory(xs, us, t, dxs, ys)


fig, axes = plt.subplots(
    3, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes[0].plot(t, theta, "b")
axes[0].set_ylabel("Theta", fontsize=graphical.default_fontsize)
axes[0].set_xlabel("v", fontsize=graphical.default_fontsize)
axes[0].tick_params(labelsize=graphical.default_fontsize)
axes[0].grid(True)

axes[1].plot(t, dtheta, "b")
# axes[1].plot(t, dtheta_num, "r")
axes[1].set_ylabel("w", fontsize=graphical.default_fontsize)
axes[1].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[1].tick_params(labelsize=graphical.default_fontsize)
axes[1].grid(True)

axes[2].plot(t, ddtheta, "b")
# axes[2].plot(t, ddtheta_num, "r")
axes[2].set_ylabel("dw", fontsize=graphical.default_fontsize)
axes[2].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[2].tick_params(labelsize=graphical.default_fontsize)
axes[2].grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


fig, axes = plt.subplots(
    1, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes.plot(x, y, "r")
axes.set_ylabel("y", fontsize=graphical.default_fontsize)
axes.set_xlabel("x", fontsize=graphical.default_fontsize)
axes.axis("equal")
axes.set(xlim=(-25, 25), ylim=(-25, 25))
axes.tick_params(labelsize=graphical.default_fontsize)
axes.grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


sys.traj = traj

sys.plot_trajectory("xu")
sys.animate_simulation()
