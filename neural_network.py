from functions import *
import numpy as np


# import functions

class Layer:
    INPUT, HIDDEN, OUTPUT = range(3)


class Neuron:
    no_of_inputs = 0
    inputs = None
    output = None
    weights = None
    output_func = None
    bias = 1
    layer = None

    def __init__(self, no_of_inputs, func=Sigmoid, layer=Layer.INPUT):
        if not issubclass(func, ActivationFunction):
            raise TypeError("func should inherit ActivationFunction in module functions !")
        self.no_of_inputs = no_of_inputs
        self.inputs = self.weights = np.array([1] * no_of_inputs)
        self.output_func = func
        self.layer = layer

    def calc(self, inputs):
        self.inputs = np.array(inputs)
        self.output = self.inputs.dot(self.weights) + self.bias
        self.output = self.output_func.f(self.output)
        return self.output


if __name__ == '__main__':
    n = Neuron(5, func=Sigmoid)
    # a = n.calc([1, 2, 3, 4, 5])
