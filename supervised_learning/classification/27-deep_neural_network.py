#!/usr/bin/env python3
""" Deep neural network
class for binary classification"""
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt


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
        """ Calculates the forward propagation of the neural network
            Args:
                X: numpy.ndarray with shape (nx, m) that contains
                the input data
            Returns:
                The output of the neural network and the cache, respectively
        """
        # Save the input data to the cache dictionary
        self.__cahce = {}
        self.cache['A0'] = X
        L = self.__L
        # Loop over all layers
        for l in range(1, L):
            # Calculate the net input fr the current layer
            W_key = 'W' + str(l + 1)
            b_key = 'b' + str(l + 1)
            A_key = 'A' + str(l)

            Z = np.dot(self.__weights[W_key], self.__cache[A_key])
            Z += self.__weights[b_key]
            # Apply the sigmoid activation function
            self.__cache['A' + str(l + 1)] = 1 / (1 + np.exp(-Z))
        # Calculate the net input for the output layer
        W_key = 'W' + str(L + 1)
        b_key = 'b' + str(L + 1)
        A_key = 'A' + str(L)

        Z = np.matmul(self.__weights[W_key], self.__cache[A_key] + \
                      self.__weights[b_key])
        # Return the output of the neural network and the cache
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates cost of the model using logistic regression """
        # Number of examples
        m = Y.shape[1]
        # Compute the cost
        logprob = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprob)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates neural network's predictions """
        A, _ = self.forward_prop(X)
        # Calculate the cost
        cost = self.cost(Y, A)
        # Apply the sigmoid activation function
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient
            descent """
        # Number of examples in input data
        m = Y.shape[1]
        dZ = cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            # Get the cached activations
            A_prev = cache['A' + str(i - 1)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ_step1 = np.dot(self.weights['W' + str(i)].T, dZ)
            dZ = dZ_step1 * (A_prev * (1 - A_prev))
            # Update the weights and biases
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ trains model """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for k in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if k % step == 0 or k == iterations:
                cost = self.cost(Y, A)

                if verbose is True:
                    print(f'Cost after {k} iterations: {cost}')

                if graph is True:
                    costs.append(cost)

            if k < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format. """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        # Check if the file exists
        if not os.path.exists(filename):
            return None

        with open(filename, 'rb') as file:
            return pickle.load(file)
