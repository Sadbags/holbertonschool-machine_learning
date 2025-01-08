#!/usr/bin/env python3
""" module defines Neuron """
import numpy as np


class Neuron:
    """ defines a single neuron  """
    def __init__(self, nx):
        """ This function initializes the Neuron instance """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Attributes
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for the weights vector """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias """
        return self.__b

    @property
    def A(self):
        """ Getter for the activated output """
        return self.__A

    def forward_prop(self, X):
        """ This function calculates the forward propagation of the neuron """
        w_sum = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-w_sum))
        return self.__A

    def cost(self, Y, A):
        """ calculates the cost """
        m = Y.shape[1]
        logprobs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprobs)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron's predictions """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ calculates one pass """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db