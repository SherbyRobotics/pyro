###############################################################################
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
from pyro.dynamic.vehicle_dynamic import DynamicBicycle
from pyro.dynamic.vehicle_dynamic import LinearTire
from pyro.dynamic.vehicle_dynamic import Pacejka


sys = DynamicBicycle()

sys.tire_model_f = LinearTire(Ca=1000.0)  # Rigidité latérale avant réduite
sys.a = 3.5  # Distance avant-essieu plus longue


sys.tire_model_r = LinearTire(Ca=80000.0)
sys.b = 0.5  # Distance arrière-essieu plus courte

sys.x0 = np.array([0, 0, 0, 0.0, 0, 0])


def control(t):
    w_rear = 20.0
    delta = 0.4

    return np.array([w_rear, delta])


sys.t2u = control


sys.compute_trajectory()
sys.plot_trajectory("x")
sys.animate_simulation(time_factor_video=1.0)
