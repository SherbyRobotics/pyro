# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt


from pyro.analysis import graphical
from pyro.planning import trajectorygeneration
from pyro.dynamic import rigidbody
from pyro.planning import plan

###############################################################################

# fixed initial position for now
# initial angular velocity is related to jerk of trajectory
x0 = 0.0
y0 = 0.0
z0 = 0.0

# fixed final position for now
xf = 10.0
yf = 10.0
zf = np.pi 
tf = 5

ddx0 = xf * 0.1
ddy0 = 0.0# ddx0 * np.tan(z0)

ddxf = 0.0
ddyf = yf * -0.1#-ddxf * np.tan(zf)


gex = trajectorygeneration.SingleAxisPolynomialTrajectoryGenerator(poly_N=9)
gex.x0_N = 3
gex.xf_N = 3
gex.x0 = np.array([x0, 0, ddx0, 0, 0, 0])
gex.xf = np.array([xf, 0, ddxf, 0, 0, 0])
gex.poly_N = 7
gex.Ws = np.array([0.01, 1.0, 1.0, 1.0, 1.0, 1.0, .0])
px, X, t = gex.solve()
x = X[0, :]
dx = X[1, :]
ax = X[2, :]
dax = X[3, :]
ddax = X[4, :]

gey = trajectorygeneration.SingleAxisPolynomialTrajectoryGenerator(poly_N=9)
gey.x0_N = 3
gey.xf_N = 3
gey.x0 = np.array([y0, 0, ddy0, 0, 0, 0])
gey.xf = np.array([yf, 0, ddyf, 0, 0, 0])
gey.poly_N = 7
gey.Ws = np.array([0.01, 1.0, 1.0, 1.0, 1.0, 1.0, .0])
py, Y, t = gey.solve()
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
axes.set(xlim=(-5, 25), ylim=(-5, 25))
axes.tick_params(labelsize=graphical.default_fontsize)
axes.grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


sys.traj = traj

ani = sys.animate_simulation()
