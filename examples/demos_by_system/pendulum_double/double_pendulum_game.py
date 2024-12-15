# -*- coding: utf-8 -*-

import numpy as np

##############################################################################
from pyro.dynamic import pendulum
from pyro.control import nonlinear
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation

##############################################################################

sys = pendulum.DoublePendulum()

# Initial condition
sys.x0[0] = -np.pi

# Parameters
sys.d1 = 0.2
sys.d2 = 0.2
sys.I1 = 2.0
sys.I2 = 2.0

# Max/Min torque
sys.u_ub[0] = +20
sys.u_ub[1] = +20
sys.u_lb[0] = -20
sys.u_lb[1] = -20

#   Cost function
sys.cost_function.Q[0, 0] = 100.0
sys.cost_function.Q[0, 0] = 100.0
sys.cost_function.R[0, 0] = 10.0
sys.cost_function.R[0, 0] = 10.0

# Controller
planner = DirectCollocationTrajectoryOptimisation(sys, 0.2, 20)
planner.x_start = sys.x0
planner.x_goal = np.array([0, 0, 0, 0])
planner.maxiter = 500
planner.set_linear_initial_guest(True)
planner.compute_optimal_trajectory()

ctl = nonlinear.ComputedTorqueController(sys, planner.traj)
ctl.rbar = np.array([0, 0])
ctl.w0 = 5
ctl.zeta = 1


game = sys.convert_to_pygame( tf=10.0, dt=0.01, ctl=ctl, renderer="pygame" )
