###############################################################################
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
from pyro.dynamic.vehicle_dynamic import DynamicBicycle
from pyro.dynamic.vehicle_dynamic import LinearTire
from pyro.dynamic.vehicle_dynamic import Pacejka


sys = DynamicBicycle()

sys.tire_model_f = Pacejka(B=10.0, C=1.3, D=1.0, E=0.97)
sys.tire_model_r = LinearTire(Ca=20000.0)


sys.x0 = np.array([0, 0, 0.0, 0.0, 0, 0])


def control_law(t):

    delta = +0.2
    w = 3.0 * t

    # if t > 2.0:
    #     w = 40.0
    #     delta = 0.0

    # if t > 6.0:
    #     w = 40.0
    #     delta = -0.2

    # if t > 12.0:
    #     w = 60.0
    #     delta = -0.2

    return np.array([w, delta])


sys.t2u = control_law

# Simulation
sys.compute_trajectory(20.0)

# Affichage Trajectoire
sys.plot_trajectory("x")

# Animation
sys.animate_simulation(time_factor_video=1.0)
