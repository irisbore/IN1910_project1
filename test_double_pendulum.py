import numpy as np
import pytest

from double_pendulum import DoublePendulum

dp = DoublePendulum()
omega1 = 0.15
omega2 = 0.15


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
    assert abs(dp._delta(theta1, theta2) - expected) < 1e-10


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
        abs(dp._domega1_dt(theta1, theta2, omega1, omega2) - expected)
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
        abs(dp._domega2_dt(theta1, theta2, omega1, omega2) - expected)
        < 1e-10
    )


def test_equilibrium():
    dp.solve([0, 0, 0, 0], 5, 0.01)
    assert np.all(dp.theta1 == 0) and np.all(
        dp.theta2 == 0) and np.all(dp.t == np.arange(0, 5 + 0.01, 0.01))


def test_radius_L1():
    dp = DoublePendulum(L1=2)
    dp.solve([np.pi / 6, 1, np.pi / 3, 2], 5, 0.02)
    assert np.all((dp.x1)**2 + (dp.y1)**2 == pytest.approx((dp.L1) ** 2))


def test_radius_L2():
    dp = DoublePendulum(L2=2)
    dp.solve([np.pi / 6, 1, np.pi / 3, 2], 5, 0.02)
    assert np.all((dp.x2 - dp.x1)**2 + (dp.y2 - dp.y1)**2 ==
                  pytest.approx((dp.L2) ** 2))
