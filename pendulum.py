from math import radians

import matplotlib.pyplot as plt
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
        acceleration of gravity (default: 9.81)

    Attributes
    ----------
    L : int or float, optional
        length of rod (default: 1)
    M : int or float, optional
        mass of pendulum (default: 1)
    g : int or float, optional
        acceleration of gravity (default: 9.81)
    t : array
        time array of solution
    theta : array
        solution for theta (position) for each value in t
    omega : array
        solution for omega for each value in t
    x : array
        x coordinate of pendulum for each value in time
    y : array
        y coordinate of pendulum for each value in time
    potential : array
        potential enrgy of pendulum for each value in time
    vx : array
        the gradient of x for each value in time
    vy : array
        the gradient of y for each value in time
    kinetic : array
        kinetic energy of pendulum for each value in time
    """

    def __init__(self, L=1, M=1, g=9.81):
        self.L = L
        self.M = M
        self.g = g

    def __call__(self, t, y):
        """
        Returns the RHS of the system of equations

        ...
        Parameters
        ----------
        t : int or float
                time
        y : 2-tuple (float, float)
            theta - the angle, omega - the angular velocity


        Returns
        -------
        2-tuple (float, float)
            the derivative of theta, the derivative of omega
        """
        return y[1], -self.g / self.L * np.sin(y[0])

    def solve(self, y0, T, dt, angles="rad"):
        """
        Solves the ODE for a given initial condition and time (0, T], and stores the solution

        ...
        Parameters
        ----------
        y0: array_like, (float, float)
            initial condition with (theta, omega)
        T: int or float
            time stop, upper time where solver evaluates
        dt: int or float
            time steps the solver should evaluate
        angels: string, optional
            "rad" (default): the initial conditions are given in radians
            "deg": the initial conditions are given in degrees
        """
        if angles == "deg":
            y0 = np.radians(y0)
        t = np.arange(0, T + dt, dt)
        self._sol = solve_ivp(self.__call__, (0, T), y0, t_eval=t)

    @property
    def t(self):
        try:
            return self._sol.t
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta(self):
        try:
            return self._sol.y[0]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def omega(self):
        try:
            return self._sol.y[1]
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
        return 0.5 * self.M * (self.vx ** 2 + self.vy ** 2)


class DampenedPendulum(Pendulum):
    """
    Subclass that represents the damping of a Pendulum

    ...

    Parameters
    ----------
    B = int or float, optional
        dampening parameter - how fast the system loses energy
    L : int or float, optional
        length of rod (default: 1)
    M : int or float, optional
        mass of pendulum (default: 1)
    g : int or float, optional
        acceleration of gravity (default: 9.81)

    Attributes
    ----------
    B = int or float, optional
        dampening parameter - how fast the system loses energy
    L : int or float, optional
        length of rod (default: 1)
    M : int or float, optional
        mass of pendulum (default: 1)
    g : int or float, optional
        acceleration of gravity (default: 9.81)
    """

    def __init__(self, B, L=1, M=1, g=9.81):
        self.B = B
        super().__init__(L, M, g)

    def __call__(self, t, y):
        """
        Returns the RHS of the system of equations

        ...
        Parameters
        ----------
        t : int or float
                time
        y : 2-tuple (float, float)
            theta - the angle, omega - the angular velocity


        Returns
        -------
        2-tuple (float, float)
            the derivative of theta, the derivative of omega
        """
        return y[1], -self.g / self.L * np.sin(y[0]) - self.B / self.M * y[1]


if __name__ == "__main__":
    p = Pendulum()
    p.solve([np.pi / 6, 2], 5, 0.01)
    plt.plot(
        p.t, p.theta, p.t, p.potential, p.t, p.kinetic, p.t, p.potential + p.kinetic
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    d = DampenedPendulum(2)
    d.solve([np.pi / 6, 2], 5, 0.01)
    plt.plot(d.t, d.kinetic + d.potential)
    plt.show()
