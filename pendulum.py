import numpy as np


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
