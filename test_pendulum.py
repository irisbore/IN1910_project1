import pendulum
import pytest
import numpy as np


@pytest.mark.parametrize(
    "arg, expected_output",
    [[(np.pi / 6, 0.15), (0.15, -9.81 / 2.7 * 0.5)], [(0, 0), (0, 0)]],
)
def test_call(arg, expected_output):
    # with pytest.raises(ValueError):
    # pendulum.__call__(arg)
    p = pendulum.Pendulum(L=2.7)
    assert p(t=0, y=arg) == pytest.approx(expected_output)
