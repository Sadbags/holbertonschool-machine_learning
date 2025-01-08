#!/usr/bin/env python3
""" This module defines class for binary classification"""
import numpy as np


class NeuralNetwork:
    """ This class defines a neural network """
    def __init__(self, nx, nodes):
        """ Class constructor """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if isinstance(nodes, int) is False:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Initialize weights and biases
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ weights vector of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """ This method retrieves the bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """ This method retrieves the activated output of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """ This method retrieves the weights vector of the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """ This retrieves the bias of the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """ This retrieves the activated output of the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """ This method calculates the forward propagation
            of the neural network """
        # Calculate the hidden layer
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calculate the output neuron
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ This calculates the cost of the model using
            logistic regression
        """
        m = Y.shape[1]
        logprobs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprobs)
        return cost

    def evaluate(self, X, Y):
        """ This method evaluates the neural network """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        return prediction, cost
