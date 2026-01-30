###############################################################################
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
from pyro.analysis import graphical
from pyro.dynamic import mechanical
from pyro.dynamic import rigidbody
from pyro.kinematic import geometry
from pyro.kinematic import drawing


##############################################################################
# Tire Models
##############################################################################


class TireModel:
    """Base Strategy for Tire-Road Interaction"""

    def get_forces(self, alpha, kappa, Fz):
        raise NotImplementedError


class LinearTire(TireModel):
    def __init__(self, Ca=60000, Ck=100000, mu=1.0):
        self.Ca, self.Ck = Ca, Ck
        self.mu = mu

    def get_forces(self, alpha, kappa, Fz):
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


class Pacejka94Tire(TireModel):
    """Pacejka 'Magic Formula' (B, C, D, E)"""

    def __init__(self, B=10.0, C=1.3, D=1.0, E=0.97):
        self.B, self.C, self.D, self.E = B, C, D, E

    def get_forces(self, alpha, kappa, Fz):
        # TODO: Implement Pacejka formula
        return 0.0, 0.0


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
        self.a = 1.2
        self.b = 1.6
        self.L = self.a + self.b
        self.R_wheel = 0.3  # Rayon de la roue [m]

        # Masse / Inertie
        self.mass = 1500.0
        self.inertia = 2500.0

        # Paramètres Physiques
        self.gravity = 9.81
        self.rho = 1.225
        self.CdA = 0.3 * 2.2

        # Modèles de Pneus
        self.tire_model = LinearTire()

        # Graphique
        self.wheel_len = 0.6
        self.wheel_width = 0.2

    ###########################################################################
    def compute_tire_physics(self, v_body, u_inputs):
        """
        Calcul les forces des pneus
        v_body : [u, v, r]
        u : [w_rear, delta]
        """
        u = v_body[0]
        v = v_body[1]
        r = v_body[2]

        w_rear = u_inputs[0]  # Vitesse de rotation commandée [rad/s]
        delta = u_inputs[1]  # Angle de braquage [rad]

        # 1. Vitesses linéaires aux moyeux
        vx_f = u
        vy_f = v + self.a * r

        vx_r = u
        vy_r = v - self.b * r

        # 2. Glissement Latéral (Alpha)
        alpha_f = delta - np.arctan2(vy_f, vx_f)
        alpha_r = 0.0 - np.arctan2(vy_r, vx_r)

        # 3. Glissement Longitudinal (Kappa)
        kappa_f = 0
        kappa_r = (w_rear * self.R_wheel - vx_r) / vx_r

        # 4. Charges Verticales (Statique pas de transfert de charge)
        Fz_f = self.mass * self.gravity * (self.b / self.L)
        Fz_r = self.mass * self.gravity * (self.a / self.L)

        # 5. Calcul des Forces Pneus (Pacejka gère Fx et Fy)
        Fx_f_tire, Fy_f_tire = self.tire_model.get_forces(alpha_f, kappa_f, Fz_f)
        Fx_r_tire, Fy_r_tire = self.tire_model.get_forces(alpha_r, kappa_r, Fz_r)

        return Fx_f_tire, Fy_f_tire, Fx_r_tire, Fy_r_tire

    ###########################################################################
    def d(self, q, v, u):

        # Forces Pneus
        Fx_f, Fy_f, Fx_r, Fy_r = self.compute_tire_physics(v, u)
        delta = u[1]

        # Projection dans le repère châssis
        c_d, s_d = np.cos(delta), np.sin(delta)

        # Forces Totales sur le châssis
        # Axe x (Force propulsion arrière + composante freinage braquage avant)
        Sum_Fx = (Fx_f * c_d - Fy_f * s_d) + Fx_r

        # Axe y
        Sum_Fy = (Fx_f * s_d + Fy_f * c_d) + Fy_r

        # Moment z
        Sum_Mz = self.a * (Fx_f * s_d + Fy_f * c_d) - self.b * Fy_r

        # Ajout Traînée Aérodynamique
        F_aero = 0.5 * self.rho * self.CdA * v[0] * abs(v[0])
        Sum_Fx -= F_aero

        F_ext_body = np.array([Sum_Fx, Sum_Fy, Sum_Mz])

        return -F_ext_body

    #############################
    def u2e(self, u):
        return np.zeros(self.dof)

    ###########################################################################
    def B(self, q, u):
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
    sys.x0 = np.array([0, 0, 0.0, 2, 0, 0])

    def control_law(t):
        w = 20.0
        delta = 0.0

        if t > 4.0:
            delta = 0.02 * (t - 4.0)  # Braquage à gauche

        if t > 10.0:
            w = 20 + 20 * (t - 10.0)

        if t > 12.0:
            delta = delta - 0.05 * (t - 12.0)

        return np.array([w, delta])

    sys.t2u = control_law

    # Simulation
    sys.compute_trajectory(20.0)

    # Affichage Trajectoire
    sys.plot_trajectory("x")

    # Animation
    sys.animate_simulation(time_factor_video=1.0)
