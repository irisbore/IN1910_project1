from math import radians

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pendulum import Pendulum


class DoublePendulum:
    """
    Class that represents a double pendulum

    Parameters
    ----------
    L1 : int or float, optional
        length of rod 1 (default: 1)
    L2: int or float, optional
        length of rod 2 (default: 1)
    M1 : int or float, optional
        mass of pendulum 1 (default: 1)
    M2 : int or float, optional
        mass of pendulum 2 (default: 1)
    G : int or float, optional
        gravity constant (default: 9.81)
    """

    def __init__(self, M1=1, M2=1, L1=1, L2=1, G=9.81):
        self.M1, self.M2, self.L1, self.L2, self.G = M1, M2, L1, L2, G

    def _delta(self, theta1, theta2):
        return theta2 - theta1

    def _domega1_dt(self, theta1, theta2, omega1, omega2):
        num = (
            self.M2
            * self.L1
            * (omega1) ** 2
            * np.sin(self._delta(theta1, theta2))
            * np.cos(self._delta(theta1, theta2))
            + self.M2 * self.G * np.sin(theta2) *
            np.cos(self._delta(theta1, theta2))
            + self.M2 * self.L2 * (omega2) ** 2 *
            np.sin(self._delta(theta1, theta2))
            - (self.M1 + self.M2) * self.G * np.sin(theta1)
        )
        den = (self.M1 + self.M2) * self.L1 - self.M2 * self.L1 * (
            np.cos(self._delta(theta1, theta2))
        ) ** 2
        return num / den

    def _domega2_dt(self, theta1, theta2, omega1, omega2):
        num = (
            -self.M2
            * self.L2
            * omega2 ** 2
            * np.sin(self._delta(theta1, theta2))
            * np.cos(self._delta(theta1, theta2))
            + (self.M1 + self.M2)
            * self.G
            * np.sin(theta1)
            * np.cos(self._delta(theta1, theta2))
            - (self.M1 + self.M2)
            * self.L1
            * omega1 ** 2
            * np.sin(self._delta(theta1, theta2))
            - (self.M1 + self.M2) * self.G * np.sin(theta2)
        )
        den = (self.M1 + self.M2) * self.L2 - self.M2 * self.L2 * (
            np.cos(self._delta(theta1, theta2))
        ) ** 2
        return num / den

    def __call__(self, t, y):
        """
        Returns the RHS of the system of equations

        ...
        Parameters
        ----------
        t : int or float
                time
        y : 4-tuple (float, float, float, float)
            theta1 - the angle of pendulum 1,
            omega1 - the angular velocity of pendulum 1,
            theta2 - the angle of pendulum 2,
            omega2 - the angular velocity of pendulum 2


        Returns
        -------
        4-tuple (float, float, float, float)
            the derivative of theta1,
            the derivative of omega1,
            the derivative of theta2,
            the derivative of omega2
        """
        return (
            y[1],
            self._domega1_dt(y[0], y[2], y[1], y[3]),
            y[3],
            self._domega2_dt(y[0], y[2], y[1], y[3]),
        )

    def solve(self, y0, T, dt, angles="rad"):
        """
        Solves the ODE for a given initial condition and time (0, T], and stores the solution

        ...
        Parameters
        ----------
        y0: array_like, (float, float, float, float)
            initial condition with (theta1, omega1, theta2, omega2)
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
        self.dt = dt
        self._sol = solve_ivp(self.__call__, (0, T), y0,
                              t_eval=t, method="Radau")

    def create_animation(self):
        """Creates animation"""
        # Create empty figure
        fig = plt.figure()

        # Configure figure
        plt.axis("equal")
        plt.axis((-3, 3, -3, 3))

        # Make an "empty" plot object to be updated throughout the animation
        (self.pendulums,) = plt.plot([], [], "o-", lw=2)

        # Call FuncAnimation
        self.animation = animation.FuncAnimation(
            fig,
            self._next_frame,
            frames=np.linspace(0, len(self.x1) - 1, 600, dtype=int),
            repeat=None,
            interval=1000 * self.dt,
            blit=True,
        )

    def _next_frame(self, i):
        """Returns the next frame to plot in the animation"""
        self.pendulums.set_data(
            (0, self.x1[i], self.x2[i]), (0, self.y1[i], self.y2[i])
        )
        return (self.pendulums,)

    def show_animation(self):
        """Shows the animation (need to call create_animation first!)"""
        plt.show()

    def save_animation(self, filename):
        """Saves the animation """
        self.animation.save(filename="double_pendulum.mp4", fps=60)

    @property
    def t(self):
        try:
            return self._sol.t
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta1(self):
        try:
            return self._sol.y[0]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta2(self):
        try:
            return self._sol.y[2]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def x1(self):
        return self.L1 * np.sin(self.theta1)

    @property
    def y1(self):
        return -self.L1 * np.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self.L2 * np.sin(self.theta2)

    @property
    def y2(self):
        return self.y1 - self.L2 * np.cos(self.theta2)

    @property
    def potential(self):
        P1 = self.M1 * self.G * (self.y1 + self.L1)
        P2 = self.M2 * self.G * (self.y2 + self.L1 + self.L2)
        return P1 + P2

    @property
    def vx1(self):
        return np.gradient(self.x1, self.t)

    @property
    def vy1(self):
        return np.gradient(self.y1, self.t)

    @property
    def vx2(self):
        return np.gradient(self.x2, self.t)

    @property
    def vy2(self):
        return np.gradient(self.y2, self.t)

    @property
    def kinetic(self):
        K1 = 0.5 * self.M1 * (self.vx1 ** 2 + self.vy1 ** 2)
        K2 = 0.5 * self.M2 * (self.vx2 ** 2 + self.vy2 ** 2)
        return K1 + K2


def labels():
    plt.xlabel("x")
    plt.ylabel("y")


if __name__ == "__main__":
    dp = DoublePendulum()
    dp.solve([np.pi / 6, 2, 0, 0], 10, 0.01)
    plt.plot(dp.t, dp.potential, dp.t, dp.kinetic,
             dp.t, dp.potential + dp.kinetic)
    dp.create_animation()
    labels()
    plt.show()
    # dp.save_animation("example_simulation.mp4")
    dp.solve([np.pi / 6 + 0.2, 2, 0.5, 1], 5, 0.01)
    plt.plot(dp.x1, dp.y1, "co", markersize=0.2,
             label="Inner pendulum for ic1")
    plt.plot(dp.x2, dp.y2, "c", label="Outer pendulum for ic1")
    dp.solve([np.pi / 6 + 0.4, 2 + 0.2, 0.7, 1.3], 5, 0.01)
    plt.plot(dp.x1, dp.y1, "yo", markersize=0.2,
             label="Inner pendulum for ic2")
    plt.plot(dp.x2, dp.y2, "y", label="Outer pendulum for ic2")
    dp.solve([np.pi / 6 + 0.6, 2 + 0.4, 1, 1.6], 5, 0.01)
    plt.plot(dp.x1, dp.y1, "go", markersize=0.2,
             label="Inner pendulum for ic3")
    plt.plot(dp.x2, dp.y2, "g", label="Outer pendulum for ic3")
    plt.legend()
    labels()
    plt.savefig("chaotic_pendulum.png")
    plt.show()
