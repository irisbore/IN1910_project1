import test_pendulum
import numpy as np
import pytest


G = 9.81
M1 = 1
M2 = 1
L1 = 1
L2 = 1
omega1 = 0.15
omega2 = 0.15


def delta(theta1, theta2):
    return theta2 - theta1


def domega1_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
    num = (
        M2
        * L1
        * (omega1) ** 2
        * np.sin(delta(theta1, theta2))
        * np.cos(delta(theta1, theta2))
        + M2 * G * np.sin(theta2) * np.cos(delta(theta1, theta2))
        + M2 * L2 * (omega2) ** 2 * np.sin(delta(theta1, theta2))
        - (M1 + M2) * G * np.sin(theta1)
    )
    den = (M1 + M2) * L1 - M2 * L1 * (np.cos(delta(theta1, theta2))) ** 2
    return num / den


def domega2_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
    num = (
        -M2
        * L2
        * omega2 ** 2
        * np.sin(delta(theta1, theta2))
        * np.cos(delta(theta1, theta2))
        + (M1 + M2) * G * np.sin(theta1) * np.cos(delta(theta1, theta2))
        - (M1 + M2) * L1 * omega1 ** 2 * np.sin(delta(theta1, theta2))
        - (M1 + M2) * G * np.sin(theta2)
    )
    den = (M1 + M2) * L2 - M2 * L2 * (np.cos(delta(theta1, theta2))) ** 2
    return num / den


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5235987755982988, 0.5235987755982988),
        (0.5235987755982988, 0, -0.5235987755982988),
        (0.5235987755982988, 0.5235987755982988, 0.0),
    ],
)
def test_delta(theta1, theta2, expected):
    assert abs(delta(theta1, theta2) - expected) < 1e-10


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5235987755982988, 3.4150779130841977),
        (0.5235987755982988, 0, -7.864794228634059),
        (0.5235987755982988, 0.5235987755982988, -4.904999999999999),
    ],
)
def test_domega1_dt(theta1, theta2, expected):
    assert (
        abs(domega1_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2) - expected)
        < 1e-10
    )


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5235987755982988, -7.8737942286340585),
        (0.5235987755982988, 0, 6.822361597534335),
        (0.5235987755982988, 0.5235987755982988, 0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    assert (
        abs(domega2_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2) - expected)
        < 1e-10
    )
