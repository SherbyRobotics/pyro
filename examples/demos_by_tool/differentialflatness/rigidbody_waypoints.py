# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt


from pyro.analysis import graphical
from pyro.planning import trajectorygeneration
from pyro.dynamic import rigidbody
from pyro.planning import plan

###############################################################################
# Waypoints
xyt = np.array([[ 0.0,  0.0, 0.0],
                [10.0,  0.0, 5.0], 
                [10.0, 10.0, 10.0], 
                [ 0.0, 10.0, 15.0]]).T



xtraj = trajectorygeneration.MultiPointSingleAxisPolynomialTrajectoryGenerator(
        poly_N=9,
        diff_N=7,
        con_N=4,
        x0=np.array([xyt[0,0], 0.0, 0.0, 0.0]),
        xf=np.array([xyt[0,3], 0.0, 0.0, 0.0]),
        tc=np.array([xyt[2,0], xyt[2,1], xyt[2,2], xyt[2,3]]),
        xc=np.array([[xyt[0,1], xyt[0,2]]]),
        dt=0.01,
    )

xtraj.Ws[0]= 0.01
xtraj.Ws[1]= 1.0
xtraj.Ws[2]= 1.0
xtraj.Ws[3]= 1.0
xtraj.Ws[4]= 1.0

b, A, p, X, t = xtraj.solve()

x = X[0, :]
dx = X[1, :]
ax = X[2, :]
dax = X[3, :]
ddax = X[4, :]

ytraj = trajectorygeneration.MultiPointSingleAxisPolynomialTrajectoryGenerator(
        poly_N=9,
        diff_N=7,
        con_N=4,
        x0=np.array([xyt[1,0], 0.0, 0.0, 0.0]),
        xf=np.array([xyt[1,3], 0.0, 0.0, 0.0]),
        tc=np.array([xyt[2,0], xyt[2,1], xyt[2,2], xyt[2,3]]),
        xc=np.array([[xyt[1,1], xyt[1,2]]]),
        dt=0.01,
    )

ytraj.Ws[0]= 0.01
ytraj.Ws[1]= 1.0
ytraj.Ws[2]= 1.0
ytraj.Ws[3]= 1.0
ytraj.Ws[4]= 1.0

b, A, p, Y, t = ytraj.solve()

y = Y[0, :]
dy = Y[1, :]
ay = Y[2, :]
day = Y[3, :]
dday = Y[4, :]

# Position theta
theta = np.arctan2(ay, ax)
# theta = np.arctan( (ay/ax))
s = np.sin(theta)
c = np.cos(theta)
dtheta = (day * c - dax * s) / (ay * s + ax * c)  # TODO check analytical equation, seems wrong
ddtheta = (
    s * (-ddax + ax * dtheta**2 - 2 * day * dtheta)
    + c * (dday - ay * dtheta**2 - 2 * dax * dtheta)
) / (
    ax * c + ay * s
)  # TODO check analytical equation, seems wrong

dtheta_num = np.diff(theta, n=1, prepend=0.0)
ddtheta_num = np.diff(dtheta, n=1, prepend=0.0)

# dtheta = dtheta_num
# ddtheta = ddtheta_num

# Create traj
steps = len(t)
xs = np.zeros((steps, 6))
ys = np.zeros((steps, 6))
us = np.zeros((steps, 2))
dxs = np.zeros((steps, 6))

sys = rigidbody.RigidBody2D()

sys.mass = 0.8
sys.inertia = 1.0
sys.l_t = 1.0

m = sys.mass
J = sys.inertia
r = sys.l_t

x_cg = x - J / (m * r) * np.cos(theta)
y_cg = y - J / (m * r) * np.sin(theta)

xs[:, 0] = x_cg
xs[:, 1] = y_cg
xs[:, 2] = theta

M = np.array([[m, 0], [0, m]])

ax_cg = (
    ax + J / (m * r) * np.sin(theta) * ddtheta + J / (m * r) * np.cos(theta) * dtheta**2
)
ay_cg = (
    ay - J / (m * r) * np.cos(theta) * ddtheta + J / (m * r) * np.sin(theta) * dtheta**2
)

# COmpute forces
for i in range(steps):
    R = np.array(
        [[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]
    )
    a_cg = np.array([ax_cg[i], ay_cg[i]])
    us[i, :] = np.linalg.inv(R) @ M @ a_cg


traj = plan.Trajectory(xs, us, t, dxs, ys)

fig, axes = plt.subplots(
    2, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes[0].plot(t, ax_cg, "b")
axes[0].set_ylabel("ax_cg", fontsize=graphical.default_fontsize)
axes[0].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[0].tick_params(labelsize=graphical.default_fontsize)
axes[0].grid(True)

axes[1].plot(t, ay_cg, "b")
axes[1].set_ylabel("ay_cg", fontsize=graphical.default_fontsize)
axes[1].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[1].tick_params(labelsize=graphical.default_fontsize)
axes[1].grid(True)


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
axes.plot(x_cg, y_cg, "b")
axes.set_ylabel("y", fontsize=graphical.default_fontsize)
axes.set_xlabel("x", fontsize=graphical.default_fontsize)
axes.axis("equal")
axes.set(xlim=(-15, 25), ylim=(-15, 25))
axes.tick_params(labelsize=graphical.default_fontsize)
axes.grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


sys.traj = traj

ani = sys.animate_simulation()
