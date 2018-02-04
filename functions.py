import numpy as np
from abc import abstractmethod, ABC


class ActivationFunction:

    @staticmethod
    @abstractmethod
    def f(x):
        pass

    @staticmethod
    @abstractmethod
    def fprime_y(y):
        pass

    @staticmethod
    @abstractmethod
    def fprime(x):
        pass


class Sigmoid(ActivationFunction):

    def f(x):
        return 1 / (1 + np.exp(-x))

    def fprime_y(y):
        return y * (1 - y)

    def fprime(x):
        return Sigmoid.fprime_y(Sigmoid.f(x))
