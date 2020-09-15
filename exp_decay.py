import numpy as np
from scipy.integrate import solve_ivp


class ExponentialDecay:
    """
    Class that represents the exponential decay ODE

    ...

    Attributes
    ----------
    a : float
        decay constant

    Methods
    -------
    solve
    """

    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        return -self.a * u

    def solve(self, u0, T, dt):
        sol = solve_ivp(self.__call__, (0, T), (u0,))
        return sol.t, sol.y[0]


if __name__ == '__main__':
    a = 0.4
    u0 = [1, 2, 3, 4]
    T = 5
    dt = 0.1
