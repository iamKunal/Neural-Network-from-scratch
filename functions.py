import numpy as np
from abc import abstractmethod, ABC


class ActivationFunction:
    '''
    Standard Activation Function Classs for the Neurons.
    '''
    @staticmethod
    @abstractmethod
    def f(x):
        '''
        Returns the value of the function at input parameter x
        :param x: Input Parameter
        :return: Function Value
        '''
        pass

    @staticmethod
    @abstractmethod
    def fprime_y(y):
        '''
        Returns the derivative of the function in terms of the function value.
        :param y: Function value
        :return: Derivative
        '''
        pass

    @staticmethod
    @abstractmethod
    def fprime(x):
        '''
        Returns the derivative of the function in terms of the input value to the function.
        :param x: Input Value
        :return: Derivative
        '''
        pass


class Sigmoid(ActivationFunction):
    '''
    The standard sigmoid function : 1/(1 + exp(-x))
    '''
    def f(x):
        return 1 / (1 + np.exp(-x))

    def fprime_y(y):
        return y * (1 - y)

    def fprime(x):
        return Sigmoid.fprime_y(Sigmoid.f(x))
