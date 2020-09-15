import pytest

import exp_decay


def test_call():
    assert exp_decay.ExponentialDecay(0.4)(0, 3.2) == pytest.approx(-1.28)
