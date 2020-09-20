from math import radians

import numpy as np
from scipy.integrate import solve_ivp


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

    def solve(self, y0, T, dt, angles="rad"):
        """"""
        if angles == "deg":
            y0 = [radians(y0[0]), radians(y0[1])]
        t = np.arange(0, T + dt, dt)
        self.sol = solve_ivp(self.__call__, (0, T), y0, t_eval=t)

    @property
    def t(self):
        try:
            return self.sol.t
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta(self):
        try:
            return self.sol.y[0]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def omega(self):
        try:
            return self.sol.y[1]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def x(self):
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        return -self.L * np.cos(self.theta)

    @property
    def potential(self):
        return self.M * self.g * (self.y + self.L)

    @property
    def vx(self):
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        return np.gradient(self.y, self.t)

    @property
    def kinetic(self):
        return 1 / 2 * self.M * (self.vx**2 + self.vy**2)


if __name__ == "__main__":
    p = Pendulum()
    p.solve([4, 5], 5, 0.01)
    p.t
    print(p.kinetic)
