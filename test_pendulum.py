import pendulum
import pytest
import numpy as np


@pytest.mark.parametrize(
    "arg, expected_output",
    [[(np.pi / 6, 0.15), (0.15, -9.81 / 2.7 * 0.5)], [(0, 0), (0, 0)]],
)
def test_call(arg, expected_output):
    p = pendulum.Pendulum(L=2.7)
    assert p(t=0, y=arg) == pytest.approx(expected_output)


@pytest.mark.parametrize("arg", ["t", "theta", "omega"])
def test_property(arg):
    p = pendulum.Pendulum()
    with pytest.raises(AttributeError):
        p.eval(arg)


def test_equilibrium():
    p = pendulum.Pendulum()
    p.solve([0, 0], 5, 0.01)
    assert (
        np.all(p.theta == 0)
        and np.all(p.omega == 0)
        and (p.t == np.arange(0, 5 + 0.01, 0.01)).all()
    )
