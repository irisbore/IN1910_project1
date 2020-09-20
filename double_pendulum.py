import numpy as np

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
        den = (self.M1 + self.M2) * self.L1 - self.M2 * \
            self.L1 * (np.cos(self._delta(theta1, theta2))) ** 2
        return num / den

    def _domega2_dt(self, theta1, theta2, omega1, omega2):
        num = (
            -self.M2
            * self.L2
            * omega2 ** 2
            * np.sin(self._delta(theta1, theta2))
            * np.cos(self._delta(theta1, theta2))
            + (self.M1 + self.M2) * self.G * np.sin(theta1) *
            np.cos(self._delta(theta1, theta2))
            - (self.M1 + self.M2) * self.L1 * omega1 ** 2 *
            np.sin(self._delta(theta1, theta2))
            - (self.M1 + self.M2) * self.G * np.sin(theta2)
        )
        den = (self.M1 + self.M2) * self.L2 - self.M2 * \
            self.L2 * (np.cos(self._delta(theta1, theta2))) ** 2
        return num / den

    def __call__(self, t, y):
        return y[1], self._domega1_dt(y[0], y[2], y[1], y[3]), y[3], self._domega2_dt(y[0], y[2], y[1], y[3])


if __name__ == '__main__':
    dp = DoublePendulum()
    dp(0, [np.pi / 6, 2, np.pi / 6, 1])
