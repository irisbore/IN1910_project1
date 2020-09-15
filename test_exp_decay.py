import pytest
import exp_decay


def test_call():
    assert exp_decay.ExponentialDecay(0.4)(0, 3.2) == pytest.approx(-1.28)


def test_solve():
    a, u0, T, dt = 0.4, 1, 7, 0.5
    decay_model = exp_decay.ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)
