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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
