import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class ExponentialDecay:
    """
    Class that represents the exponential decay ODE

    ...

    Parameters
    ----------
    a : float
        decay constant


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
        t : int or float
                time
        u : float
                constant

        Returns
        -------
        float
            the derivative of u
        """
        return -self.a * u

    def solve(self, u0, T, dt):
        """
        Solves the ODE u'(t) = -au for a given initial condition

        ...
        Parameters
        ----------
        u0: int or float
            initial condition
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
        t_eval = np.arange(0, T + dt, dt)
        sol = solve_ivp(self.__call__, (0, T), (u0,), t_eval=t_eval)
        return sol.t, sol.y.ravel()


if __name__ == "__main__":
    a = 0.4
    u0 = 0.1
    T = 5
    dt = 0.1
    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)
    plt.plot(t, u)
    plt.show()
