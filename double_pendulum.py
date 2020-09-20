from pendulum import Pendulum


class DoublePendulum:
    g = 9.81

    def __init__(self, M1=1, M2=1, L1=1, L2=1):
        self.pendulum1 = Pendulum(L1, M1)
        self.pendulum2 = Pendulum(L2, M2)

    def delta(theta1, theta2):
        return theta2 - theta1

    def domega1_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
        num = (
            M2
            * L1
            * (omega1) ** 2
            * np.sin(self.delta(theta1, theta2))
            * np.cos(self.delta(theta1, theta2))
            + M2 * G * np.sin(theta2) * np.cos(self.delta(theta1, theta2))
            + M2 * L2 * (omega2) ** 2 * np.sin(self.delta(theta1, theta2))
            - (M1 + M2) * G * np.sin(theta1)
        )
        den = (M1 + M2) * L1 - M2 * L1 * (np.cos(delta(theta1, theta2))) ** 2
        return num / den

    def domega2_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
        num = (
            -M2
            * L2
            * omega2 ** 2
            * np.sin(self.delta(theta1, theta2))
            * np.cos(self.delta(theta1, theta2))
            + (M1 + M2) * G * np.sin(theta1) * np.cos(self.delta(theta1, theta2))
            - (M1 + M2) * L1 * omega1 ** 2 * np.sin(self.delta(theta1, theta2))
            - (M1 + M2) * G * np.sin(theta2)
        )
        den = (M1 + M2) * L2 - M2 * L2 * (np.cos(delta(theta1, theta2))) ** 2
        return num / den
