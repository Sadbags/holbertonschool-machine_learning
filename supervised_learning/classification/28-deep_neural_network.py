#!/usr/bin/env python3
""" This module defines a deep neural network
class for binary classification"""
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle


class DeepNeuralNetwork:
    """This class defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """ Class constructor """
        # Check if nx is an integer, if not raise a TypeError
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Check if nx is a positive integer, if not raise a ValueError
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        # Initialize number of layers
        self.__L = len(layers)
        # Initialize cache as an empty dictionary
        self.__cache = {}
        # Initialize weights as an empty dictionary
        self.__weights = {}
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        for i in range(self.L):
            if i == 0:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], nx
                ) * np.sqrt(2 / nx)
            else:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
            # Initialize biases to 0's fr each layer
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """This method retrieves the number of layers"""
        return self.__L

    @property
    def cache(self):
        """This method retrieves the intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """This method retrieves the weights and biases"""
        return self.__weights

    @property
    def activation(self):
        """This method retrieves the activation function"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network
        """
        # Store the input data in the cache
        self.__cache["A0"] = X

        # Get the total number of layers in the network
        L1 = self.__L

        # Loop over all layers in the network, excluding the output layer
        for lopper in range(1, L1):
            Z = (
                np.matmul(
                    self.__weights["W" + str(lopper)],
                    self.__cache["A" + str(lopper - 1)],
                )
                + self.__weights["b" + str(lopper)]
            )

            if self.__activation == "sig":
                # Sigmoid activation function
                A = 1 / (1 + np.exp(-Z))
            else:
                # Hyperbolic tangent activation function
                A = np.tanh(Z)

            # Store the output of the activation function in the cache
            self.__cache["A" + str(lopper)] = A

        Z = (
            np.matmul(self.__weights["W" + str(L1)],
                      self.__cache["A" + str(L1 - 1)])
            + self.__weights["b" + str(L1)]
        )

        # Apply the softmax function to the weighted input of the output layer
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)

        # Store the output of the softmax function in the cache
        self.__cache["A" + str(L1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """ Calculate the cross-entropy cost for multiclass classification """
        # Number of examples
        m = Y.shape[1]
        log_loss = -(1 / m) * np.sum(Y * np.log(A))

        return log_loss

    def evaluate(self, X, Y):
        """ Evaluate the network's predictions """
        # Run forward propagation to get the activated output of the network
        A, _ = self.forward_prop(X)

        # Calculate the cost of the network
        cost = self.cost(Y, A)

        predictions = np.where(A == np.max(A, axis=0), 1, 0)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient
        descent on the neural network """
        # Number of examples in input data
        m = Y.shape[1]
        # Calculate the gradients of the output data
        dZ = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            # Get the cached activations
            A_prev = cache["A" + str(i - 1)]
            # Calculate the derivatives of the weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ_step1 = np.dot(self.weights["W" + str(i)].T, dZ)
            if self.__activation == "sig":
                dZ = dZ_step1 * (A_prev * (1 - A_prev))
            elif self.__activation == "tanh":
                dZ = dZ_step1 * (1 - (A_prev ** 2))
            # Update the weights and biases
            self.weights["W" + str(i)] -= alpha * dW
            self.weights["b" + str(i)] -= alpha * db

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """ Trains the deep neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize cost list for graphing
        costs = []

        # Train the network
        for i in range(iterations + 1):
            # Perform forward propagation and calculate the cost
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # Plot the training cost
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Returns the evaluation of the training data
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as file:
            return pickle.load(file)
