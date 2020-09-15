import pendulum
import pytest
import numpy as np


def test_call():
    #with pytest.raises(ValueError):
    #pendulum.__call__(arg)
    p = pendulum.Pendulum(L=2.7)
    assert p(t=0, y=(np.pi/6, 0.15)) == pytest.approx((0.15, -9.81/2.7*0.5))
