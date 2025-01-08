#!/usr/bin/env python3
""" Deep neural network
class for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """ This class defines a deep neural network
        performing binary classification"""
    def __init__(self, nx, layers):
        """ Class constructor """
        # Checks if nx is an integer, if not raise a TypeError
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Checks if nx is a positive integer if not raise a ValueError
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        # Initialize cache as an empty dictionary
        self.__cache = {}
        # Initialize weights as an empty dictionary
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2/layers[i - 1])
            # Initialize biases to 0's fr each layer
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Retrieves the number of layers"""
        return self.__L

    @property
    def cache(self):
        """ Retrieves the intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """ Retrieves the weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """ calculates the forward propagation """
        self.__cache['A0'] = X
        A = X
        for k in range(1, self.__L + 1):
            # add the weight of each node
            W = self.__weights['W' + str(k)]
            # add the biases
            b = self.__weights['b' + str(k)]
            # use matmul to calculate the output neuron
            z = np.matmul(W, A) + b
            # calculate the sigmoid function on every layer
            A = 1 / (1 + np.exp(-z))
            self.__cache['A' + str(k)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ Calculates cost of the model using logistic regression"""
        # Number of the examples
        m = Y.shape[1]
        # Computes the cost
        logprob = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprob)
        return cost
