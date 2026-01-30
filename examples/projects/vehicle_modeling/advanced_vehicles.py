# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:45:37 2018

@author: Alexandre
"""


###############################################################################
import numpy as np

##############################################################################
from pyro.dynamic import system
from pyro.dynamic import vehicle_steering

###############################################################################


# NOTE: models in development and not fully validated


###############################################################################


class LateralDynamicBicycleModelwithSpeedInput(vehicle_steering.KinematicCarModel):
    """
    Equations of Motion
    -------------------------
    dv_y    = (F_yf*cos(delta)+F_yr)/m - v_x*dtheta         "lateral force sum (F_y = m*dv_y) gives lateral acceleration ddv_y"
    ddtheta = (a*F_yf*cos(delta)-b*F_yr)/Iz                 "Torque sum at mass center (T = I*ddtheta) gives the angular acceleration of the vehicle ddtheta"
    dtheta  = dtheta                                        "dtheta is already a state which is the yaw rate"
    dX      = v_x*cos(theta)-v_y*sin(theta)                 "To obtain cartesian position"
    dY      = v_x*sin(theta)+v_y*cos(theta)

    Where
    F_yf is the lateral force applied perpendicularly with the front wheel (N)
    F_yr is the lateral force applied perpendicularly with the rear wheel (N)
    v_y  is the lateral velocity of the mass center (m/s)
    v_x  is the longitudinal velocity of the mass center (m/s)
    delta is the steering angle (rad)
    theta is the yaw angle of the vehicle (rad)
    m     is the mass of the vehicle (kg)
    Iz    is the inertia for a rotation along the z-axis (kg*m^2)
    a     is the distance from the front axle to the mass centre (m)
    b     is the distance from the rear axle to the mass centre (m)
    (X,Y) is the cartesian position of the vehicle
    """

    ############################
    def __init__(self):
        """ """

        # Dimensions
        self.n = 5
        self.m = 2
        self.p = 5

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Lateral Dynamic Bicyle Model"
        self.state_label = ["v_y", "dtheta", "theta", "X", "Y"]
        self.input_label = ["delta", "v_x"]
        self.output_label = ["v_y", "dtheta", "theta", "X", "Y"]

        # Units
        self.state_units = ["[m/s]", "[rad/s]", "[rad]", "[m]", "[m]"]
        self.input_units = ["[rad]", "[m/sec]"]
        self.output_units = ["[m/s]", "[rad/s]", "[rad]", "[m]", "[m]"]

        # Model param (used for animation purposes only change the values)
        self.width = 2.00
        self.a = 2.00
        self.b = 3.00
        self.lenght = self.a + self.b
        self.lenght_tire = 0.40
        self.width_tire = 0.15

        self.g = 9.81
        self.mass = 2000
        self.Iz = 1.00 / 12.00 * self.mass * ((self.a + self.b) ** 2 + self.width**2)

        # Graphic output parameters
        self.dynamic_domain = True
        self.dynamic_range = self.width * 2

    #######################################
    def piecewise_tiremodel(self, x, u):

        # Tire-road friction coefficient
        self.mu = 0.9
        # Compute normal forces on the tires
        F_nf = self.mass * self.g * self.b / (self.b + self.a)
        F_nr = self.mass * self.g * self.a / (self.b + self.a)
        # Compute the max forces available
        max_F_f = F_nf * self.mu
        max_F_r = F_nr * self.mu
        # Compute the lateral "slip-slope"
        self.max_alpha_stat = 0.12
        slip_ratio_f = max_F_f / (self.max_alpha_stat)
        slip_ratio_r = max_F_r / (self.max_alpha_stat)
        if u[1] == 0:
            slip_f = 0
            slip_r = 0
            F_yf = 0
            F_yr = 0
        else:
            slip_f = u[0] - np.arctan((x[0] + self.a * x[1]) / u[1])
            slip_r = np.arctan((self.b * x[1] - x[0]) / u[1])

        if slip_f < -0.12:
            F_yf = -max_F_f
        elif slip_f > 0.12:
            F_yf = max_F_f
        else:
            F_yf = slip_ratio_f * slip_f

        if slip_r < -0.12:
            F_yr = -max_F_r
        elif slip_r > 0.12:
            F_yr = max_F_r
        else:
            F_yr = slip_ratio_r * slip_r

        return F_yf, F_yr

    #############################

    def f(self, x=np.zeros(5), u=np.zeros(2), t=0):
        """
        Continuous time foward dynamics evaluation

        dx = f(x,u,t)

        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1

        OUPUTS
        dx : state derivative vectror n x 1

        """

        dx = np.zeros(self.n)  # State derivative vector
        F_yf, F_yr = self.piecewise_tiremodel(x, u)

        dx[0] = (F_yf * np.cos(u[0]) + F_yr) / self.mass - u[1] * x[1]
        dx[1] = (self.a * F_yf * np.cos(u[0]) - self.b * F_yr) / self.Iz
        dx[2] = x[1]
        dx[3] = u[1] * np.cos(x[2]) - x[0] * np.sin(x[2])
        dx[4] = u[1] * np.sin(x[2]) - x[0] * np.cos(x[2])

        return dx

    #############################
    def xut2q(self, x, u, t):
        """compute config q"""

        q = np.array([x[3], x[4], x[2], u[0]])

        return q


##############################################################################


class LateralDynamicBicycleModelwithForceInputs(
    LateralDynamicBicycleModelwithSpeedInput
):
    """
    Equations of Motion
    -------------------------

    dx   = V cos ( phi )
    dy   = V sin ( phi )
    dphi = V/l tan ( beta )

    STATES_DERIVATIVE
    dv_x    = (F_xf*cos(delta)-F_yf*sin(delta)+F_xr)/m-v_y*dtheta   "longitudinal force sum (F_x = m*dv_x) gives longitudinal acceleration dv_x"
    dv_y    = (F_yf*cos(delta)+F_xf*sin(delta)+F_yr)/m - v_x*dtheta "lateral force sum (F_y = m*dv_y) gives lateral acceleration dv_y"
    ddtheta = (a*(F_yf*cos(delta)+F_xf*sin(delta))-b*F_yr)/Iz       "Torque sum at mass center (T = I*ddtheta) gives the angular acceleration of the vehicle ddtheta"
    dtheta  = dtheta                                                "dtheta is already a state which is the yaw rate"
    dX      = v_x*cos(theta)-v_y*sin(theta)                         "To obtain cartesian position"
    dY      = v_x*sin(theta)+v_y*cos(theta)

    INPUTS
    delta is the steering angle (rad)
    F_xf is the longitudinal force given by the front propulsion (N)
    F_xr is the longitudinal force given by the rear propulsion (N)


    Where
    F_yf is the lateral force applied perpendicularly with the front wheel (N)
    F_yr is the lateral force applied perpendicularly with the rear wheel (N)
    v_y  is the lateral velocity of the mass center (m/s)
    v_x  is the longitudinal velocity of the mass center (m/s)
    delta is the steering angle (rad)
    theta is the yaw angle of the vehicle (rad)
    m     is the mass of the vehicle (kg)
    Iz    is the inertia for a rotation along the z-axis (kg*m^2)
    a     is the distance from the front axle to the mass centre (m)
    b     is the distance from the rear axle to the mass centre (m)
    (X,Y) is the cartesian position of the vehicle

    """

    ############################
    def __init__(self):
        """ """

        # Dimensions
        self.n = 6
        self.m = 3
        self.p = 6

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Full Dynamic Bicyle Model without slip"
        self.state_label = ["v_x", "v_y", "dtheta", "theta", "X", "Y"]
        self.input_label = ["delta", "F_xf", "F_xr"]
        self.output_label = ["v_x", "v_y", "dtheta", "theta", "X", "Y"]

        # Units
        self.state_units = ["[m/s]", "[m/s]", "[rad/s]", "[rad]", "[m]", "[m]"]
        self.input_units = ["[rad]", "[N]", "[N]"]
        self.output_units = ["[m/s]", "[m/s]", "[rad/s]", "[rad]", "[m]", "[m]"]

        # Model param (used for animation purposes only change the values)
        self.width = 2.00
        self.a = 2.00
        self.b = 3.00
        self.lenght = self.a + self.b
        self.lenght_tire = 0.40
        self.width_tire = 0.15

        self.g = 9.81
        self.mass = 1000
        self.Iz = 1.00 / 12.00 * self.mass * ((self.a + self.b) ** 2 + self.width**2)

        # Graphic output parameters
        self.dynamic_domain = True
        self.dynamic_range = 5

    #############################
    def piecewise_tiremodel(self, x, u):

        # Tire-road friction coefficient
        self.mu = 0.9
        F_nf = self.mass * self.g * self.b / (self.b + self.a)
        F_nr = self.mass * self.g * self.a / (self.b + self.a)
        # Compute the max forces available
        max_F_f = F_nf * self.mu
        max_F_r = F_nr * self.mu
        # Compute the lateral "slip-slope"
        self.max_alpha_stat = 0.12
        slip_ratio_f = max_F_f / (self.max_alpha_stat)
        slip_ratio_r = max_F_r / (self.max_alpha_stat)
        if x[0] == 0:
            slip_f = 0
            slip_r = 0
        elif x[0] > 0:
            slip_f = np.arctan((x[1] + self.a * x[2]) / x[0]) - u[0]
            slip_r = np.arctan((x[1] - self.b * x[2]) / x[0])
        else:
            slip_f = -(np.arctan((x[1] + self.a * x[2]) / x[0]) + u[0])
            slip_r = -np.arctan((x[1] - self.b * x[2]) / x[0])
        if slip_f < -0.12:
            F_yf = max_F_f
        elif slip_f > 0.12:
            F_yf = -max_F_f
        else:
            F_yf = -slip_ratio_f * slip_f

        if slip_r < -0.12:
            F_yr = max_F_r
        elif slip_r > 0.12:
            F_yr = -max_F_r
        else:
            F_yr = -slip_ratio_r * slip_r

        return F_yf, F_yr

    #############################

    def f(self, x=np.zeros(6), u=np.zeros(3), t=0):
        """
        Continuous time foward dynamics evaluation

        dx = f(x,u,t)

        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1

        OUPUTS
        dx : state derivative vectror n x 1

        """
        dx = np.zeros(self.n)  # State derivative vector
        F_yf, F_yr = self.piecewise_tiremodel(x, u)

        dx[0] = (u[1] * np.cos(u[0]) - F_yf * np.sin(u[0]) + u[2]) / self.mass - x[
            1
        ] * x[2]
        dx[1] = (F_yf * np.cos(u[0]) + u[1] * np.sin(u[0]) + F_yr) / self.mass - x[
            0
        ] * x[2]
        dx[2] = (
            self.a * (F_yf * np.cos(u[0]) + u[1] * np.sin(u[0])) - self.b * F_yr
        ) / self.Iz
        dx[3] = x[2]
        dx[4] = x[0] * np.cos(x[3]) - x[1] * np.sin(x[3])
        dx[5] = x[0] * np.sin(x[3]) + x[1] * np.cos(x[3])

        return dx

    #############################
    def xut2q(self, x, u, t):
        """compute config q"""

        q = np.array([x[4], x[5], x[3], u[0]])

        return q


##############################################################################


class FullDynamicBicycleModelwithTorquesInputs(
    LateralDynamicBicycleModelwithSpeedInput
):
    """
    Equations of Motion
    -------------------------
    STATES_DERIVATIVE
    dv_x    = (F_xf*cos(delta)-F_yf*sin(delta)+F_xr)/m-v_y*dtheta   "longitudinal force sum (F_x = m*dv_x) gives longitudinal acceleration dv_x"
    dv_y    = (F_yf*cos(delta)+F_xf*sin(delta)+F_yr)/m - v_x*dtheta "lateral force sum (F_y = m*dv_y) gives lateral acceleration dv_y"
    ddtheta = (a*(F_yf*cos(delta)+F_xf*sin(delta))-b*F_yr)/Iz       "Torque sum at mass center (T = I*ddtheta) gives the angular acceleration of the vehicle ddtheta"
    dtheta  = dtheta                                                "dtheta is already a state which is the yaw rate"
    dX      = v_x*cos(theta)-v_y*sin(theta)                         "To obtain cartesian position"
    dY      = v_x*sin(theta)+v_y*cos(theta)
    domega_f = (T_F-F_xf*R)/I_propf
    domega_r = (T_r-F_xr*R)/I_propr

    ***** DEPENDS DU MODELE DE TIRE COMPLET EX: S=omega/v_x et F_x,F_y = f(F_z,alpha,S et mu) *****

    INPUTS
    delta is the steering angle (rad)
    T_f is the front wheel torque (N)
    T_r is the rear wheel torque (N)


    Where
    F_yf is the lateral force applied perpendicularly with the front wheel (N)
    F_yr is the lateral force applied perpendicularly with the rear wheel (N)
    v_y  is the lateral velocity of the mass center (m/s)
    v_x  is the longitudinal velocity of the mass center (m/s)
    delta is the steering angle (rad)
    theta is the yaw angle of the vehicle (rad)
    m     is the mass of the vehicle (kg)
    Iz    is the inertia for a rotation along the z-axis (kg*m^2)
    a     is the distance from the front axle to the mass centre (m)
    b     is the distance from the rear axle to the mass centre (m)
    (X,Y) is the cartesian position of the vehicle
    """

    ############################
    def __init__(self):
        """ """

        # Dimensions
        self.n = 8
        self.m = 3
        self.p = 8

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Full Dynamic Bicyle Model with slip"
        self.state_label = [
            "v_x",
            "v_y",
            "dtheta",
            "theta",
            "X",
            "Y",
            "omega_f",
            "omega_r",
        ]
        self.input_label = ["delta", "T_f", "T_r"]
        self.output_label = [
            "v_x",
            "v_y",
            "dtheta",
            "theta",
            "X",
            "Y",
            "omega_f",
            "omega_r",
        ]

        # Units
        self.state_units = [
            "[m/s]",
            "[m/s]",
            "[rad/s]",
            "[rad]",
            "[m]",
            "[m]",
            "[rad/s]",
            "[rad/s]",
        ]
        self.input_units = ["[rad]", "[Nm]", "[Nm]"]
        self.output_units = [
            "[m/s]",
            "[m/s]",
            "[rad/s]",
            "[rad]",
            "[m]",
            "[m]",
            "[rad/s]",
            "[rad/s]",
        ]

        # Model param (used for animation purposes only change the values)
        self.width = 2.00
        self.a = 2.00
        self.b = 3.00
        self.lenght = self.a + self.b
        self.lenght_tire = 0.40
        self.width_tire = 0.15

        self.g = 9.81
        self.mass = 1000
        self.Iz = 1.00 / 12.00 * self.mass * ((self.a + self.b) ** 2 + self.width**2)
        self.Iprop_f = 20
        self.Iprop_r = 20
        self.r = 0.3

        # Graphic output parameters
        self.dynamic_domain = True
        self.dynamic_range = 5

    #############################
    def combinedSlipTireModel(self, x, u):

        # Tire-road friction coefficient
        self.mu = 0.1
        F_nf = self.mass * self.g * self.b / (self.b + self.a)
        F_nr = self.mass * self.g * self.a / (self.b + self.a)
        # Compute the max forces available
        max_F_f = F_nf * self.mu
        max_F_r = F_nr * self.mu
        # Compute the lateral "slip-slope"
        self.max_alpha_stat = 0.12
        slip_ratio_f = max_F_f / (self.max_alpha_stat)
        slip_ratio_r = max_F_r / (self.max_alpha_stat)
        # Simulation of road-tire behavior according to slip-ratios
        self.max_slip_stat = 0.03
        slip_slope_f = max_F_f / self.max_slip_stat
        slip_slope_r = max_F_r / self.max_slip_stat
        # Compute slip ratios and longitudinal forces for both tires (depend on Vx (x[0]), omega_f (x[6]), omega_r (x[7]))
        if x[0] == 0 or x[6] == 0:
            long_slip_f = 0
            F_xf = 0
        elif u[1] == 0:
            long_slip_f = 0
            F_xf = 0
        else:
            long_slip_f = (x[6] * self.r - x[0]) / np.maximum(
                np.absolute(x[0]), np.absolute(x[6] * self.r)
            )
            if long_slip_f < -0.03:
                F_xf = -max_F_f
            elif long_slip_f > 0.03:
                F_xf = max_F_f
            else:
                F_xf = slip_slope_f * long_slip_f
        if x[0] == 0 and x[7] == 0:
            F_xr = 0
            long_slip_r = 0
        elif u[2] == 0:
            long_slip_r = 0
            F_xr = 0
        else:
            long_slip_r = (x[7] * self.r - x[0]) / max(x[0], x[7] * self.r)
            if long_slip_r < -0.03:
                F_xr = -max_F_r
            elif long_slip_r > 0.03:
                F_xr = max_F_r
            else:
                F_xr = slip_slope_r * long_slip_r
        # Compute slip angles and lateral forces for both tires (depends on Vx (x[0]),Vy (x[1]) and delta (u[0]))
        if x[0] == 0:
            slip_f = 0
            slip_r = 0
        elif x[0] > 0:
            slip_f = np.arctan((x[1] + self.a * x[2]) / x[0]) - u[0]
            slip_r = np.arctan((x[1] - self.b * x[2]) / x[0])
        else:
            slip_f = -(np.arctan((x[1] + self.a * x[2]) / x[0]) - u[0])
            slip_r = -np.arctan((x[1] - self.b * x[2]) / x[0])
        if slip_f < -0.12:
            F_yf = max_F_f
        elif slip_f > 0.12:
            F_yf = -max_F_f
        else:
            F_yf = -slip_ratio_f * slip_f

        if slip_r < -0.1:
            F_yr = max_F_r
        elif slip_r > 0.1:
            F_yr = -max_F_r
        else:
            F_yr = -slip_ratio_r * slip_r

        return F_yf, F_yr, F_xf, F_xr

    #############################

    def f(self, x=np.zeros(8), u=np.zeros(3), t=0):
        """
        Continuous time foward dynamics evaluation

        dx = f(x,u,t)

        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1

        OUPUTS
        dx : state derivative vectror n x 1

        """
        dx = np.zeros(self.n)  # State derivative vector
        F_yf, F_yr, F_xf, F_xr = self.combinedSlipTireModel(x, u)

        dx[0] = (F_xf * np.cos(u[0]) - F_yf * np.sin(u[0]) + F_xr) / self.mass - x[
            1
        ] * x[2]
        dx[1] = (F_yf * np.cos(u[0]) + F_xf * np.sin(u[0]) + F_yr) / self.mass - x[
            0
        ] * x[2]
        dx[2] = (
            self.a * (F_yf * np.cos(u[0]) + F_xf * np.sin(u[0])) - self.b * F_yr
        ) / self.Iz
        dx[3] = x[2]
        dx[4] = x[0] * np.cos(x[3]) - x[1] * np.sin(x[3])
        dx[5] = x[0] * np.sin(x[3]) + x[1] * np.cos(x[3])
        dx[6] = (u[1] - self.r * F_xf) / self.Iprop_f
        dx[7] = (u[2] - self.r * F_xr) / self.Iprop_r

        return dx

    ############################
    def xut2q(self, x, u, t):
        """compute config q"""

        q = np.array([x[4], x[5], x[3], u[0]])

        return q


##############################################################################


class FullDynamicBicycleModelwithVoltInput(LateralDynamicBicycleModelwithSpeedInput):
    """
    Equations of Motion
    -------------------------
    STATES_DERIVATIVE
    dv_x    = (F_xf*cos(delta)-F_yf*sin(delta)+F_xr)/m - v_y*dtheta   "longitudinal force sum (F_x = m*dv_x) gives longitudinal acceleration dv_x"
    dv_y    = (F_yf*cos(delta)+F_xf*sin(delta)+F_yr)/m - v_x*dtheta "lateral force sum (F_y = m*dv_y) gives lateral acceleration dv_y"
    ddtheta = (a*(F_yf*cos(delta)+F_xf*sin(delta))-b*F_yr)/Iz       "Torque sum at mass center (T = I*ddtheta) gives the angular acceleration of the vehicle ddtheta"
    dtheta  = dtheta                                                "dtheta is already a state which is the yaw rate"
    dX      = v_x*cos(theta)-v_y*sin(theta)                         "To obtain cartesian position"
    dY      = v_x*sin(theta)+v_y*cos(theta)
    domega_r = ((V-omega_r*gR*k)*gR*k/R - F_xr*r - B(omega))/J      "Propulsion model"

    ***** DEPENDS DU MODELE DE TIRE COMPLET EX: S=omega/v_x et F_x,F_y = f(F_z,alpha,S et mu) *****

    INPUTS
    delta is the steering angle (rad)
    propulsion voltage (V)


    Where
    F_yf is the lateral force applied perpendicularly with the front wheel (N)
    F_yr is the lateral force applied perpendicularly with the rear wheel (N)
    v_y  is the lateral velocity of the mass center (m/s)
    v_x  is the longitudinal velocity of the mass center (m/s)
    delta is the steering angle (rad)
    theta is the yaw angle of the vehicle (rad)
    m     is the mass of the vehicle (kg)
    Iz    is the inertia for a rotation along the z-axis (kg*m^2)
    a     is the distance from the front axle to the mass centre (m)
    b     is the distance from the rear axle to the mass centre (m)
    (X,Y) is the cartesian position of the vehicle
    r     is the wheels radius (m)
    gR    is the gear ratio
    R     is the motor's resistance (ohm)
    k     is the motor constant (Nm/A)
    J     is the propulsion's inertia
    B(omega) is the viscuous friction function
    """

    ############################
    def __init__(self):
        """ """

        # Dimensions
        self.n = 7
        self.m = 2
        self.p = 7

        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)

        # Labels
        self.name = "Full Dynamic Bicyle Model with slip"
        self.state_label = ["v_x", "v_y", "dtheta", "theta", "X", "Y", "omega_r"]
        self.input_label = ["delta", "V"]
        self.output_label = ["v_x", "v_y", "dtheta", "theta", "X", "Y", "omega_r"]

        # Units
        self.state_units = [
            "[m/s]",
            "[m/s]",
            "[rad/s]",
            "[rad]",
            "[m]",
            "[m]",
            "[rad/s]",
        ]
        self.input_units = ["[rad]", "[Nm]", "[Nm]"]
        self.output_units = [
            "[m/s]",
            "[m/s]",
            "[rad/s]",
            "[rad]",
            "[m]",
            "[m]",
            "[rad/s]",
        ]

        # Model param (used for animation purposes only change the values)
        self.width = 0.2
        self.a = 0.204
        self.b = 0.136
        self.lenght = self.a + self.b
        self.lenght_tire = 0.11
        self.width_tire = 0.04

        self.g = 9.81
        self.mass = 4.1
        self.Iz = 1.00 / 12.00 * self.mass * ((self.a + self.b) ** 2 + self.width**2)
        self.r = 0.055

        # Propulsion param
        self.R = 0.63
        self.k = 0.002791
        self.J = 0.120881
        self.gR = 15.3

        # Graphic output parameters
        self.dynamic_domain = True
        self.dynamic_range = 5

    #############################
    def combinedSlipTireModel(self, x, u):

        # Tire-road friction coefficient
        self.mu = 0.45
        F_nf = self.mass * self.g * self.b / (self.b + self.a)
        F_nr = self.mass * self.g * self.a / (self.b + self.a)
        # Compute the max forces available
        max_F_f = F_nf * self.mu
        max_F_r = F_nr * self.mu
        # Compute the lateral "slip-slope"
        self.max_alpha_lin = 0.12
        slip_ratio_f = -max_F_f / (self.max_alpha_lin)
        slip_ratio_r = -max_F_r / (self.max_alpha_lin)
        # Simulation of road-tire behavior according to slip-ratios
        self.max_slip_lin = 0.07
        slip_slope_r = max_F_r / self.max_slip_lin
        # Compute slip ratios and longitudinal forces for both tires (depend on Vx (x[0]), omega_f (x[6]), omega_r (x[7]))
        F_xf = 0
        if abs(x[0]) < 0.1 and abs(x[6]) < 0.1:
            F_xr = 0
            long_slip_r = 0
        else:
            long_slip_r = (x[6] * self.r - x[0]) / max(abs(x[0]), abs(x[6] * self.r))
            if long_slip_r < -self.max_slip_lin:
                F_xr = -max_F_r
            elif long_slip_r > self.max_slip_lin:
                F_xr = max_F_r
            else:
                F_xr = slip_slope_r * long_slip_r

        # Compute slip angles and lateral forces for both tires (depends on Vx (x[0]),Vy (x[1]) and delta (u[0]))
        if abs(x[0]) < 0.1:
            slip_f = 0
            slip_r = 0
        else:
            slip_f = np.arctan((x[1] + self.a * x[2]) / x[0]) - u[0]
            slip_r = np.arctan((x[1] - self.b * x[2]) / x[0])

        if slip_f < -self.max_alpha_lin:
            F_yf = max_F_f
        elif slip_f > self.max_alpha_lin:
            F_yf = -max_F_f
        else:
            F_yf = slip_ratio_f * slip_f

        if slip_r < -self.max_alpha_lin:
            F_yr = max_F_r
        elif slip_r > self.max_alpha_lin:
            F_yr = -max_F_r
        else:
            F_yr = slip_ratio_r * slip_r

        return F_yf, F_yr, F_xf, F_xr

    #############################

    def f(self, x=np.zeros(7), u=np.zeros(2), t=0):
        """
        Continuous time foward dynamics evaluation

        dx = f(x,u,t)

        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1

        OUPUTS
        dx : state derivative vectror n x 1

        """
        dx = np.zeros(self.n)  # State derivative vector
        F_yf, F_yr, F_xf, F_xr = self.combinedSlipTireModel(x, u)

        dx[0] = (F_xf * np.cos(u[0]) - F_yf * np.sin(u[0]) + F_xr) / self.mass + x[
            1
        ] * x[2]
        dx[1] = (F_yf * np.cos(u[0]) + F_xf * np.sin(u[0]) + F_yr) / self.mass - x[
            0
        ] * x[2]
        dx[2] = (
            self.a * (F_yf * np.cos(u[0]) + F_xf * np.sin(u[0])) - self.b * F_yr
        ) / self.Iz
        dx[3] = x[2]
        dx[4] = x[0] * np.cos(x[3]) - x[1] * np.sin(x[3])
        dx[5] = x[0] * np.sin(x[3]) + x[1] * np.cos(x[3])
        dx[6] = (
            (u[1] - x[6] * self.gR * self.k) * self.gR * self.k / self.R - F_xr * self.r
        ) / self.J

        return dx

    ############################
    def xut2q(self, x, u, t):
        """compute config q"""

        q = np.array([x[4], x[5], x[3], u[0]])

        return q


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    sys = LateralDynamicBicycleModelwithSpeedInput()

    def t2u(t):
        """input trajectory function"""
        delta = 0.1 * t
        v_x = 10.0 + 0.05 * t
        return np.array([delta, v_x])

    sys.t2u = t2u
    # sys.ubar = np.array([1.1, 10])

    sys.compute_trajectory(20)
    sys.plot_trajectory()
    sys.animate_simulation()

    # sys      = LateralDynamicBicycleModelwithForceInputs()
    # sys.ubar = np.array([0.8,000,200])

    # sys.plot_trajectory()
    # sys.animate_simulation()

    # sys      = FullDynamicBicycleModelwithTorquesInputs()
    # sys.ubar = np.array([0.2,0,1000])

    # sys.plot_trajectory()
    # sys.animate_simulation()
