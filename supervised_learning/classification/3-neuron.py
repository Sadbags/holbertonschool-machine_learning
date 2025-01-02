#!/usr/bin/env python3
""" module for neuron class"""

import numpy as np


class Neuron:
    """ class to create a neuron """
    def __init__(self, nx):
        """ initializes the neuron """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """ weights vector attributes """
        return self.__W

    @property
    def b(self):
        """ for the bias """
        return self.__b

    def A(self):
        """ activated output """
        return self.__A

    def forward_prop(self, X):
        """ calculates the forward propagation """
        w_sum = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-w_sum))
        return self.__A

    def cost(self, Y, A):
        """ calculates de cost of the model """
        m = Y.shape[1]
        logprobs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprobs)
        return cost
