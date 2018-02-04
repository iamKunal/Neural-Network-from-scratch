from functions import *
import numpy as np
from multiprocessing import Pool
from functools import partial
import pickle
from time import time


# import functions

class Layer:
    INPUT, HIDDEN, OUTPUT = range(3)


class Neuron:
    '''
    This is the Neuron class to be used in the Network.
    '''
    no_of_inputs = 0
    inputs = None
    output = None
    weights = None
    output_func = None
    layer = None
    target = None
    error = None

    def __init__(self, no_of_inputs, func=Sigmoid, layer=Layer.INPUT):
        '''

        :param no_of_inputs: The number of inputs to the neuron (excluding bias).
        :param func: The activation function to be used.
        :param layer: The type of layer (INPUT, HIDDEN or OUTPUT)
        '''
        if not issubclass(func, ActivationFunction):
            raise TypeError("func should inherit ActivationFunction in module functions !")
        self.no_of_inputs = no_of_inputs + 1
        self.inputs = self.weights = np.array([1] * self.no_of_inputs)
        self.output_func = func
        self.layer = layer

    def calc(self, inputs):
        '''
        Calculates the output value from the given inputs
        :param inputs: Accepts input as a list
        :return: returns the output
        '''
        self.inputs = np.array(inputs + [1])
        self.output = self.inputs.dot(self.weights)
        self.output = self.output_func.f(self.output)
        return self.output

    def calc_error(self):
        '''
        Calculated the error if the neuron is in the OUTPUT Layer
        :return:
        '''
        if not self.output:
            raise ValueError("Please call calc(), before calling calc_error().")
        if self.layer == Layer.OUTPUT:
            self.error = self.target - self.output
        return self.error


class Network:
    '''
    This is the Network class, contains a list of layers, each layer containing a list of Neurons.
    '''

    def __init__(self, layers, no_of_inputs, func=None):
        '''

        :param layers: A list containing the number of neurons in each layer
        :param no_of_inputs: The number of inputs to the first layer (ie, the number of features)
        :param func: The list of activation functions to be used at each layer
        '''
        self.layers = []
        self.no_of_inputs = 0
        self.pool_size = 1
        self.learning_rate = 0.50
        if len(layers) < 2:
            raise ValueError("Atleast 2 Layers Required !")
        if func is None:
            func = [Sigmoid] * len(layers)
        self.no_of_inputs = no_of_inputs
        self.layers.append([Neuron(self.no_of_inputs, func[0], Layer.INPUT) for _ in range(layers[0])])
        for i in range(1, len(layers) - 1):
            self.layers.append([Neuron(layers[i - 1], func[i], Layer.HIDDEN) for _ in range(layers[i])])
        self.layers.append([Neuron(layers[-2], func[-1], Layer.OUTPUT) for _ in range(layers[-1])])
        self.threads(1)

    def threads(self, pool_size):
        '''

        :param pool_size: Enter the pool size for multithreading. Defaults to one. Not much helpful if number of neurons
        per layer is too less.
        :return:
        '''
        self.pool_size = pool_size

    @staticmethod
    def single_forward(inputs, neuron):
        '''
        A helper function be used with threading that calculates the forward propagation of the neuron.
        :param inputs: Inputs to the neuron (either features or from previous layer)
        :param neuron: The neuron to be worked upon
        :return: Returns the neuron with the calculated values.
        '''
        neuron.calc(inputs)
        return neuron

    def forward(self, inputs, targets):
        '''
        Forward Propagation with multithreading.
        :param inputs: The initial inputs (features) to the Neural Network.
        :param targets: The target values (optional), used while training
        :return:
        '''
        p = Pool(self.pool_size)
        runner = partial(self.single_forward, inputs)
        out = p.map(runner, self.layers[0])
        self.layers[0] = out
        if targets is not None:
            for i in range(len(self.layers[-1])):
                self.layers[-1][i].target = targets[i]
        out = [o.output for o in out]
        # print('0 = ', out)
        for i in range(1, len(self.layers)):
            runner = partial(self.single_forward, out)
            out = p.map(runner, self.layers[i])
            self.layers[i] = out
            out = [o.output for o in out]
            # print(i, '= ', out)
        p.close()

    @staticmethod
    def single_calc_errors_output(neuron):
        '''
        A helper function be used with threading that calculates the error of the neurons in OUTPUT Layer.
        :param neuron: The neuron itself.
        :return: The neuron with calculated error is returned.
        '''
        neuron.calc_error()
        return neuron

    @staticmethod
    def single_calc_errors_not_output(errors, neuron):
        '''
        A helper function be used with threading that calculates the error of the neurons in INPUT and HIDDEN Layers.
        :param errors: The errors of the output layer.
        :param neuron: The neuron itself.
        :return: The neuron with calculated error is returned.
        '''
        error = 0.0
        for e in errors:
            error += np.sum(neuron.weights * e)
        error = error * neuron.output_func.fprime_y(neuron.output)
        neuron.error = error
        return neuron

    def calc_error(self):
        '''
        Error Calculation with Multithreading.
        :return:
        '''
        p = Pool(self.pool_size)
        neurons = p.map(self.single_calc_errors_output, self.layers[-1])
        self.layers[-1] = neurons
        errors = [neuron.error for neuron in self.layers[-1]]
        runner = partial(self.single_calc_errors_not_output, errors)
        for i in range(len(self.layers) - 1):
            ns = p.map(runner, self.layers[i])
            self.layers[i] = ns
        p.close()

    @staticmethod
    def update_weight_single(learning_rate, neuron):
        '''
        Updates weight of a single neuron, given a learning rate.
        :param learning_rate: The learning rate to be used.
        :param neuron: The neuron itself.
        :return: The neuron with new calculated weight is returned.
        '''
        neuron.weights = neuron.weights + learning_rate * neuron.error * neuron.inputs
        return neuron

    def update_weights(self, learning_rate=None):
        '''
        Weight Updatation with Multithreading.
        :param learning_rate: The learning rate to be used, defaults to 0.5
        :return:
        '''
        if learning_rate is None:
            learning_rate = self.learning_rate
        p = Pool(self.pool_size)

        runner = partial(self.update_weight_single, learning_rate)
        for i in range(len(self.layers)):
            self.layers[i] = p.map(runner, self.layers[i])
        p.close()

    def single_train(self, inputs, targets, learning_rate):
        '''
        Train the Network for a single row of dataset
        :param inputs: The inputs (features)
        :param targets: The target output values.
        :param learning_rate: The learning rate, defaults to 0.5
        :return: The current accuracy is returned. [1-|target-output|]
        '''
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.forward(inputs, targets)
        self.calc_error()
        self.update_weights()
        current_accuracy = [1 - abs(targets[i] - self.layers[-1][i].output) for i in range(len(self.layers[-1]))]
        print("Current Accuracy :", current_accuracy)
        return current_accuracy

    def single_predict(self, inputs, targets=None):
        '''
        Predict the output values for a single row of dataset.
        :param inputs: The inputs (features)
        :param targets: The target output values (optional) for accuracy measurement.
        :return: Returns the accuracy if the target output values are provided.
        '''
        self.forward(inputs, targets)
        if targets is None:
            outputs = [neuron.output for neuron in self.layers[-1]]
            return outputs
        else:
            current_accuracy = [1 - abs(targets[i] - self.layers[-1][i].output) for i in range(len(self.layers[-1]))]
            print("Current Accuracy :", current_accuracy)
            return current_accuracy

    def save(self, filename):
        '''
        Save the current Neural Network Model (with trained weights) into the file 'filename.pkl'
        :param filename: The filename to save the model to.
        :return:
        '''
        with open(filename + '.pkl', 'wb') as output:
            pickle.dump(self, output)

    @staticmethod
    def load(filename):
        '''
        Load a Neural Network Model (with trained weights) from the specified file.
        :param filename: The filename to load the model from.
        :return: Returns the loaded model.
        '''
        obj = None
        with open(filename, 'rb') as input:
            obj = pickle.load(input)
        return obj


if __name__ == '__main__':
    net = Network([2, 3, 1], 2)
    net.threads(1)
    data_set = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
                [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
                [7.673756466, 3.508563011, 1]]
    inputs = [[data[0], data[1]] for data in data_set]
    targets = [[data[2]] for data in data_set]
    a = time()
    for _ in range(150):
        print("Epoch", _)
        for i in range(len(data_set)):
            net.single_train(inputs[i], targets[i], 0.50)
    print(time() - a)
    net.save("my_network")
    # net = Network.load('my_network.pkl')
    # for i in range(len(data_set)):
    #     net.single_predict(inputs[i], targets[i])
