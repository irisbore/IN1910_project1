import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class ExponentialDecay:
    """
    Class that represents the exponential decay ODE

    ...

    Attributes
    ----------
    a : float
        decay constant

    """

    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        """
        Returns the RHS of the ODE

        ...
        Parameters
        ----------
        t: array
            time
        u: callable or float
            constant

        Returns
        -------
        float
            the derivative of u
        """
        if callable(u):
            return -self.a * u(t)
        return -self.a * u

    def solve(self, u0, T, dt):
        """
        Solves the ODE for a given initial condition

        ...
        Parameters
        ----------
        u0:

        T: int or float
            time stop
        dt: int or float
            steps

        Returns
        -------
        array
            timepoints
        array
            solution points
        """
        t = np.arange(0, T + dt, dt)
        sol = solve_ivp(self.__call__, (0, T), (u0,), t_eval=t)
        return sol.t, sol.y.ravel()


if __name__ == '__main__':
    a = 0.4
    u0 = 0.1
    T = 5
    dt = 0.1
    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)
    print(t, u)
    plt.plot(t, u)
    plt.show()
