import numpy as np

from pendulum import Pendulum
from math import radians


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
            + self.M2 * self.G * np.sin(theta2) * np.cos(self._delta(theta1, theta2))
            + self.M2 * self.L2 * (omega2) ** 2 * np.sin(self._delta(theta1, theta2))
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
            y0 = [radians(y0[0]), radians(y0[1]), radians(y0[2]), radians(y0[3])]
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


if __name__ == "__main__":
    dp = DoublePendulum()
    dp(0, [np.pi / 6, 2, np.pi / 6, 1])
