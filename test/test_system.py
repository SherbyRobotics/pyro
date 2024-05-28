import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pyro.dynamic import ContinuousDynamicSystem  # Replace 'your_module' with the actual module name

matplotlib.use('Template')
# Mock subclass to implement the f method
class MockContinuousDynamicSystem(ContinuousDynamicSystem):
    def f(self, x, u, t):
        # Simple dynamic equation for testing
        return -x + u

# Test cases
@pytest.fixture
def system():
    return MockContinuousDynamicSystem(n=2, m=2, p=2)

@pytest.fixture
def system_3d():
    return MockContinuousDynamicSystem(n=3, m=2, p=2)

def test_initialization(system):
    assert system.n == 2
    assert system.m == 2
    assert system.p == 2
    assert system.state_label == ["State 0", "State 1"]
    assert system.input_label == ["Input 0", "Input 1"]
    assert system.output_label == ["Output 0", "Output 1"]

def test_f_method(system):
    x = np.array([1.0, 2.0])
    u = np.array([0.5, 1.5])
    t = 0
    dx = system.f(x, u, t)
    assert np.array_equal(dx, -x + u)

def test_h_method(system):
    x = np.array([1.0, 2.0])
    u = np.array([0.5, 1.5])
    t = 0
    y = system.h(x, u, t)
    assert np.array_equal(y, x)

def test_t2u_method(system):
    t = 0
    u = system.t2u(t)
    assert np.array_equal(u, system.ubar)

def test_isavalidstate(system):
    x_valid = np.array([0.0, 0.0])
    x_invalid = np.array([11.0, 0.0])
    assert system.isavalidstate(x_valid) is True
    assert system.isavalidstate(x_invalid) is False

def test_isavalidinput(system):
    x = np.array([0.0, 0.0])
    u_valid = np.array([0.0, 0.0])
    u_invalid = np.array([2.0, 0.0])
    assert system.isavalidinput(x, u_valid) is True
    assert system.isavalidinput(x, u_invalid) is False

def test_fsim(system):
    x = np.array([1.0, 2.0])
    t = 0
    dx = system.fsim(x, t)
    expected_dx = -x + system.ubar
    assert np.array_equal(dx, expected_dx)

def test_x_next(system):
    x = np.array([1.0, 2.0])
    u = np.array([0.5, 1.5])
    t = 0
    dt = 0.1
    steps = 1
    x_next = system.x_next(x, u, t, dt, steps)
    expected_x_next = x + dt * (-x + u)
    assert np.allclose(x_next, expected_x_next)

def test_animate_simulation(system):
    system.compute_trajectory(tf=1.0, n=11)
    animation = system.animate_simulation()
    assert animation is not None
    plt.close('all')