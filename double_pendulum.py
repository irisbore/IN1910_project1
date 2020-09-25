from math import radians

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pendulum import Pendulum


class DoublePendulum:
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
        return (
            y[1],
            self._domega1_dt(y[0], y[2], y[1], y[3]),
            y[3],
            self._domega2_dt(y[0], y[2], y[1], y[3]),
        )

    def solve(self, y0, T, dt, angles="rad"):
        """"""
        if angles == "deg":
            y0 = [radians(y0[0]), radians(y0[1]),
                  radians(y0[2]), radians(y0[3])]
        t = np.arange(0, T + dt, dt)
        self.dt = dt
        self.sol = solve_ivp(self.__call__, (0, T), y0,
                             t_eval=t, method="Radau")

    def create_animation(self):
        # Create empty figure
        fig = plt.figure()

        # Configure figure
        plt.axis("equal")
        plt.axis((-3, 3, -3, 3))

        # Make an "empty" plot object to be updated throughout the animation
        self.pendulums, = plt.plot([], [], "o-", lw=2)

        # Call FuncAnimation
        self.animation = animation.FuncAnimation(
            fig,
            self._next_frame,
            frames=range(len(self.x1)),
            repeat=None,
            interval=1000 * self.dt,
            blit=True,
        )

    def _next_frame(self, i):
        self.pendulums.set_data(
            (0, self.x1[i], self.x2[i]), (0, self.y1[i], self.y2[i])
        )
        return (self.pendulums,)

    def show_animation(self):
        plt.show()

    def save_animation(self, filename):
        self.animation.save(filename="double_pendulum.mp4", fps=60)

    @property
    def t(self):
        try:
            return self.sol.t
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta1(self):
        try:
            return self.sol.y[0]
        except AttributeError:
            raise AttributeError("You need to call solve")
        except:
            print("Something went wrong")

    @property
    def theta2(self):
        try:
            return self.sol.y[2]
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


if __name__ == "__main__":
    dp = DoublePendulum()
    dp.solve([np.pi / 6, 2, 0, 0], 10, 0.01)
    plt.plot(dp.t, dp.potential, dp.t, dp.kinetic,
             dp.t, dp.potential + dp.kinetic)
    dp.create_animation()
    # dp.save_animation("example_simulation.mp4")
    plt.show()
