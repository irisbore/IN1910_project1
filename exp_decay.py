import scipy


class ExponentialDecay:
    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        return -self.a * u


if __name__ == '__main__':
    u = ExponentialDecay(0.4)
