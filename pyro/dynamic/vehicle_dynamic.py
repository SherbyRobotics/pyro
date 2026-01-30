###############################################################################
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
from pyro.dynamic import rigidbody
from pyro.kinematic import geometry
from pyro.kinematic import drawing


##############################################################################
# Tire Models
##############################################################################


class TireModel:
    """Base Strategy for Tire-Road Interaction"""

    def __init__(self):
        self.v_min_epsilon = 0.1

    def vel2slip(self, vx, vy, w, R):
        """Compute longitudinal and lateral slip"""

        # Adjusted longitudinal velocity to avoid division by zero
        vx_adj = np.abs(vx) + self.v_min_epsilon

        # Lateral slip angle (alpha)
        alpha = -np.arctan(vy / vx_adj)

        # Longitudinal slip ratio (kappa)
        kappa = (w * R - vx) / vx_adj

        return alpha, kappa

    def slip2forces(self, alpha, kappa, Fz):
        """Convert slip values to forces using the tire model"""
        raise NotImplementedError

    def vel2forces(self, vx, vy, w, R, Fz):
        """Compute forces directly from velocities"""
        alpha, kappa = self.vel2slip(vx, vy, w, R)
        return self.slip2forces(alpha, kappa, Fz)


class LinearTire(TireModel):

    def __init__(self, Ca=60000, Ck=100000, mu=1.0):

        TireModel.__init__(self)

        self.Ca, self.Ck = Ca, Ck
        self.mu = mu

    def slip2forces(self, alpha, kappa, Fz):
        # Forces brutes
        Fx = self.Ck * kappa
        Fy = self.Ca * alpha

        # Saturation circulaire simple (Friction circle)
        F_max = self.mu * Fz
        F_total = np.sqrt(Fx**2 + Fy**2)

        if F_total > F_max:
            ratio = F_max / F_total
            Fx *= ratio
            Fy *= ratio

        return Fx, Fy


class Pacejka(TireModel):
    """Magic Formula'"""

    def __init__(self, B=10.0, C=1.3, D=1.0, E=0.97):
        TireModel.__init__(self)
        self.B, self.C, self.D, self.E = B, C, D, E

    def slip2forces(self, alpha, kappa, Fz):
        # Fonction magique
        def mf(x, fz):
            D_scaled = self.D * fz
            return D_scaled * np.sin(
                self.C
                * np.arctan(self.B * x - self.E * (self.B * x - np.arctan(self.B * x)))
            )

        Fx = mf(kappa, Fz)
        Fy = mf(alpha, Fz)

        return Fx, Fy


##############################################################################
# Dynamic Bicycle Class
##############################################################################


class DynamicBicycle(rigidbody.RigidBody2D):
    """
    Modèle Bicyclette Dynamique héritant de RigidBody2D.

    Inputs u:
    0: w_rear [rad/s] (Vitesse de rotation roue arrière)
    1: delta  [rad]   (Angle de braquage roue avant)
    """

    def __init__(self):
        # Init parent
        super().__init__(force_inputs=0, other_inputs=2)

        self.name = "Dynamic Bicycle (Wheel Speed Input)"
        self.input_label = ["w_rear", "delta"]
        self.input_units = ["[rad/s]", "[rad]"]

        # Ranges
        self.u_ub = np.array([+200.0, +0.6])
        self.u_lb = np.array([-200.0, -0.6])
        self.x_ub = np.array([+100, +100, +10, 50, 10, 5])
        self.x_lb = np.array([-100, -100, -10, -50, -10, -5])

        # Paramètres Géométriques
        self.a = 1.0
        self.b = 1.0
        self.L = self.a + self.b
        self.r_f = 0.3  # Rayon de la roue avant [m]
        self.r_r = 0.3  # Rayon de la roue arrière [m]

        # Masse / Inertie
        self.mass = 1500.0
        self.inertia = 2500.0

        # Paramètres Physiques
        self.gravity = 9.81
        self.rho = 1.225
        self.CdA = 0.3 * 2.2

        # Modèles de Pneus
        self.tire_model_f = LinearTire()
        self.tire_model_r = LinearTire()
        # self.tire_model_f = Pacejka(B=12.0, C=1.3, D=1.0, E=0.97)
        # self.tire_model_r = Pacejka(B=12.0, C=1.3, D=1.0, E=0.97)

        # Graphique
        self.wheel_len = 0.6
        self.wheel_width = 0.2

    ###########################################################################
    def compute_wheel_velocities(self, v_body, u_inputs):
        """
        Calcul les vitesses (translation du moyeux et rotation) des roues
        v_body : [u, v, r]
        u : [w_rear, delta]
        """

        u = v_body[0]  # Vitesse longitudinale [m/s] (surge)
        v = v_body[1]  # Vitesse latérale [m/s] (sway)
        r = v_body[2]  # Vitesse de lacet [rad/s] (yaw rate)

        delta = u_inputs[1]  # Angle de braquage [rad]

        # Vitesses linéaires aux moyeux, repère du véhicule
        # Avant
        vx_f_b = u
        vy_f_b = v + self.a * r
        # Arrière
        vx_r_b = u
        vy_r_b = v - self.b * r

        # Vitesses linéaires, repère des roues
        # Avant (braquée)
        c_d, s_d = np.cos(delta), np.sin(delta)
        vx_f = c_d * vx_f_b + s_d * vy_f_b
        vy_f = -s_d * vx_f_b + c_d * vy_f_b
        # Arrière (non braquée)
        vx_r = vx_r_b
        vy_r = vy_r_b

        # Calcul des vitesses de rotation des roues
        w_r = u_inputs[0]  # Vitesse de rotation commandée [rad/s]
        w_f = vx_f / self.r_f  # Roue avant libre, no slip

        return vx_f, vy_f, w_f, vx_r, vy_r, w_r

    ###########################################################################
    def compute_tire_physics(self, v_body, u_inputs):
        """
        Calcul les forces des pneus
        v_body : [u, v, r]
        u : [w_rear, delta]
        """

        # Vitesses des roues
        vx_f, vy_f, w_f, vx_r, vy_r, w_r = self.compute_wheel_velocities(
            v_body, u_inputs
        )

        # Charges Verticales (Statique pas de transfert de charge)
        Fz_f = self.mass * self.gravity * (self.b / self.L)
        Fz_r = self.mass * self.gravity * (self.a / self.L)

        # Forces Pneus (Repère Roue)
        Fx_f, Fy_f = self.tire_model_f.vel2forces(vx_f, vy_f, w_f, self.r_f, Fz_f)
        Fx_r, Fy_r = self.tire_model_r.vel2forces(vx_r, vy_r, w_r, self.r_r, Fz_r)

        return Fx_f, Fy_f, Fx_r, Fy_r

    ###########################################################################
    def d(self, q, v, u):

        # Forces Pneus
        Fx_f, Fy_f, Fx_r, Fy_r = self.compute_tire_physics(v, u)
        delta = u[1]

        # Projection des forces des roues dans le repère châssis
        c_d, s_d = np.cos(delta), np.sin(delta)
        Fx_f_b = Fx_f * c_d - Fy_f * s_d
        Fy_f_b = Fx_f * s_d + Fy_f * c_d

        Fx_r_b = Fx_r
        Fy_r_b = Fy_r

        # Forces Totales sur le châssis
        # Axe x
        Sum_Fx = Fx_f_b + Fx_r_b

        # Axe y
        Sum_Fy = Fy_f_b + Fy_r_b

        # Moment z
        Sum_Mz = self.a * Fy_f_b - self.b * Fy_r

        # Ajout Traînée Aérodynamique
        F_aero = 0.5 * self.rho * self.CdA * v[0] * abs(v[0])
        Sum_Fx -= F_aero

        F_ext_body = np.array([Sum_Fx, Sum_Fy, Sum_Mz])

        return -F_ext_body

    #############################
    def u2e(self, u):
        # No force inputs in this model
        return np.zeros(self.dof)

    ###########################################################################
    def B(self, q, u):
        # No force inputs in this model
        return np.zeros((self.dof, self.dof))

    ###########################################################################
    # GRAPHICS
    ###########################################################################

    ###########################################################################
    def forward_kinematic_domain(self, q):
        """
        Place holder graphical output ( box with a force )
        """

        l = self.l_t * 5.0

        x = q[0]
        y = q[1]
        z = 0

        if self.dynamic_domain:

            domain = [(-l + x, l + x), (-l + y, l + y), (-l + z, l + z)]  #
        else:

            domain = [(-l, l), (-l, l), (-l, l)]  #

        return domain

    ###########################################################################
    def forward_kinematic_lines(self, q):
        """Dessin du châssis"""
        lines_pts = []
        lines_style = []
        lines_color = []

        x, y, theta = q[0], q[1], q[2]
        W_T_B = geometry.transformation_matrix_2D(theta, x, y)

        # Châssis (Rectangle)
        w = self.wheel_len * 1.5
        pts_body = np.array([[self.a, 0, 0], [-self.b, 0, 0]])
        pts_W = drawing.transform_points_2D(W_T_B, pts_body)

        lines_pts.append(pts_W)
        lines_style.append("-")
        lines_color.append("k")

        return lines_pts, lines_style, lines_color

    ###########################################################################
    def forward_kinematic_lines_plus(self, x, u, t):
        """Dessin dynamique (Roues, Vecteurs)"""
        lines_pts = []
        lines_style = []
        lines_color = []

        q = x[0:3]
        v = x[3:6]

        X, Y, Theta = q[0], q[1], q[2]
        delta = u[1]

        # Calcul physique pour affichage
        Fx_f, Fy_f, Fx_r, Fy_r = self.compute_tire_physics(v, u)

        # Matrices de transfo
        W_T_B = geometry.transformation_matrix_2D(Theta, X, Y)
        B_T_Wf = geometry.transformation_matrix_2D(delta, self.a, 0)  # Roue Avant
        B_T_Wr = geometry.transformation_matrix_2D(0, -self.b, 0)  # Roue Arrière

        # 1. Dessin des Roues (Boites noires)
        wl, ww = self.wheel_len, self.wheel_width
        wheel_shape = np.array(
            [
                [wl / 2, ww / 2, 0],
                [wl / 2, -ww / 2, 0],
                [-wl / 2, -ww / 2, 0],
                [-wl / 2, ww / 2, 0],
                [wl / 2, ww / 2, 0],
            ]
        )

        pts_wf = drawing.transform_points_2D(W_T_B @ B_T_Wf, wheel_shape)
        pts_wr = drawing.transform_points_2D(W_T_B @ B_T_Wr, wheel_shape)

        lines_pts.append(pts_wf)
        lines_style.append("-")
        lines_color.append("k")
        lines_pts.append(pts_wr)
        lines_style.append("-")
        lines_color.append("k")

        # 2. Vecteurs Vitesses (Bleu)
        v_scale = 0.2
        # Vitesse au centre des roues (Body frame)
        v_f_loc = np.array([v[0], v[1] + self.a * v[2]])
        v_r_loc = np.array([v[0], v[1] - self.b * v[2]])

        pts_vf = drawing.arrow_from_components(
            v_f_loc[0] * v_scale, v_f_loc[1] * v_scale
        )
        pts_vr = drawing.arrow_from_components(
            v_r_loc[0] * v_scale, v_r_loc[1] * v_scale
        )

        # Transfo vers monde (Attention: vitesse calculée au point d'attache, pas dans le repère tourné de la roue)
        B_T_AxleF = geometry.transformation_matrix_2D(0, self.a, 0)
        B_T_AxleR = geometry.transformation_matrix_2D(0, -self.b, 0)

        lines_pts.append(drawing.transform_points_2D(W_T_B @ B_T_AxleF, pts_vf))
        lines_style.append("-")
        lines_color.append("b")

        lines_pts.append(drawing.transform_points_2D(W_T_B @ B_T_AxleR, pts_vr))
        lines_style.append("-")
        lines_color.append("b")

        # 3. Vecteurs Forces (Rouge)
        f_scale = 0.001
        # Force avant (dans le repère de la roue braquée)
        pts_Ff = drawing.arrow_from_components(Fx_f * f_scale, Fy_f * f_scale)
        lines_pts.append(drawing.transform_points_2D(W_T_B @ B_T_Wf, pts_Ff))
        lines_style.append("-")
        lines_color.append("r")

        # Force arrière
        pts_Fr = drawing.arrow_from_components(Fx_r * f_scale, Fy_r * f_scale)
        lines_pts.append(drawing.transform_points_2D(W_T_B @ B_T_Wr, pts_Fr))
        lines_style.append("-")
        lines_color.append("r")

        return lines_pts, lines_style, lines_color


#################################################################
##################          Main                         ########
#################################################################

if __name__ == "__main__":
    """MAIN TEST"""

    sys = DynamicBicycle()

    # État initial : [x, y, theta, vx, vy, r]
    # On commence avec une vitesse latérale nulle, mais une vitesse longit de 10 m/s
    sys.x0 = np.array([0, 0, 0.0, 0.0, 0, 0])

    def control_law(t):

        delta = +0.2
        w = -5.0

        if t > 2.0:
            w = 40.0
            delta = 0.0

        if t > 4.0:
            delta = 0.2 * (t - 4.0)  # Braquage à gauche

        if t > 8.0:
            delta = -0.2

        if t > 9.0:
            w = 0.0

        if t > 10.0:
            delta = -0.4
            w = -5.0

        if t > 12.0:
            delta = 0.0
            w = 50.0

        if t > 15.0:
            delta = 0.2 * (t - 15.0)  # Braquage à gauche
            w = 50.0 + 5.0 * (t - 15.0)

        return np.array([w, delta])

    sys.t2u = control_law

    # Simulation
    sys.compute_trajectory(20.0, solver="euler")

    # Affichage Trajectoire
    sys.plot_trajectory("x")

    # Animation
    sys.animate_simulation(time_factor_video=1.0)
