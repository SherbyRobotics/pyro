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

# Waypoints       x_i   y_i  t_i
xyt = np.array([[ 0.0,  0.0, 0.0],
                [10.0,  0.0, 5.0], 
                [10.0, 15.0, 10.0], 
                [ 0.0, 5.0, 15.0]]).T



traj = trajectorygeneration.MultiPointSingleAxisPolynomialTrajectoryGenerator(
        poly_N=9,
        diff_N=7,
        con_N=4,
        x0=np.array([xyt[0,0], 0.0, 0.0, 0.0]),
        xf=np.array([xyt[0,3], 0.0, 0.0, 0.0]),
        tc=np.array([xyt[2,0], xyt[2,1], xyt[2,2], xyt[2,3]]),
        xc=np.array([[xyt[0,1], xyt[0,2]]]),
        dt=0.01,
    )

traj.Ws[0]= 0.01
traj.Ws[1]= 1.0
traj.Ws[2]= 1.0
traj.Ws[3]= 1.0
traj.Ws[4]= 1.0

b, A, p, X, t = traj.solve()

x = X[0, :]
dx = X[1, :]
ax = X[2, :]
dax = X[3, :]
ddax = X[4, :]

traj = trajectorygeneration.MultiPointSingleAxisPolynomialTrajectoryGenerator(
        poly_N=9,
        diff_N=7,
        con_N=4,
        x0=np.array([xyt[1,0], 0.0, 0.0, 0.0]),
        xf=np.array([xyt[1,3], 0.0, 0.0, 0.0]),
        tc=np.array([xyt[2,0], xyt[2,1], xyt[2,2], xyt[2,3]]),
        xc=np.array([[xyt[1,1], xyt[1,2]]]),
        dt=0.01,
    )

traj.Ws[0]= 0.01
traj.Ws[1]= 1.0
traj.Ws[2]= 1.0
traj.Ws[3]= 1.0
traj.Ws[4]= 1.0

b, A, p, Y, t = traj.solve()

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
