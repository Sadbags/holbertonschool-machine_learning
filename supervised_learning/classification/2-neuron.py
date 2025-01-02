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
