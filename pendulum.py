import numpy as np

class Pendulum:
    """
    Class that represents

    y =
    """
    def __init__(self, L=1, M=1, g=9.81):
        self.L = L
        self.M = M
        self.g = g

    def __call__(self, t, y):
        return y[1], -self.g/self.L*np.sin(y[0])
