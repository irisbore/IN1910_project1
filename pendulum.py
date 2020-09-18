import numpy as np
from math import radians


class Pendulum:
    """
    Class that represents a pendulum

    Parameters
    ----------
    L : int or float, optional
        length of rod (default: 1)
    M : int or float, optional
        mass of pendulum (default: 1)
    g : int or float, optional
        gravity constant (default: 9.81)

    Attributes
    ----------
    L : int or float, optional
        length of rod (default: 1)
    M : int or float, optional
        mass of pendulum (default: 1)
    g : int or float, optional
        gravity constant (default: 9.81)
    """

    def __init__(self, L=1, M=1, g=9.81):
        self.L = L
        self.M = M
        self.g = g

    def __call__(self, t, y):
        return y[1], -self.g / self.L * np.sin(y[0])

    def solve(self, y0, T, dt, angles=rad):
        """"""
        if angles == deg:
            y0 = (radians(y0[0]), radians(y0[1]))
        t = np.arange(0, T + dt, dt)
        sol = solve_ivp(self.__call__, (0, T), (u0,), t_eval=t)
        self.t = sol.t
        self.y = sol.y.ravel()


if __name__ == "__main__":
    a = 2
