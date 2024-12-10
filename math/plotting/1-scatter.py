#!/usr/bin/env python3
"""
Scatter plot module
This module contains a function to generate a
scatter plot of men's height vs weight.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Scatter plot function
    This function generates a scatter plot of men's
    height vs weight.
    The height and weight data are generated using
    a multivariate normal distribution.
    The scatter plot is displayed with magenta markers.
    """

    # Define the mean and covariance for
    # the multivariate normal distribution
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]

    # Set the random seed for reproducibility
    np.random.seed(5)

    # Generate the height and weight data
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180  # Adjust the weight data

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Set the labels for the y-axis and x-axis
    plt.ylabel("Weight (lbs)")
    plt.xlabel("Height (in)")

    # Set the title for the plot
    plt.title("Men's Height vs Weight")

    # Create the scatter plot with magenta markers
    plt.scatter(x, y, c="magenta")

    # Display the plot
    plt.show()
