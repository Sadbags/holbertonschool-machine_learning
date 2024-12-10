#!/usr/bin/env python3
"""Plot x ↦ y as a scatter plot."""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Scatter plot function.

    This function generates a scatter plot of men's height vs. weight.
    """
    # Define mean and covariance matrix
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]

    # Generate data for height and weight
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180  # Adjust the weight data

    # Create a figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Set axis labels
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Set plot title
    plt.title("Men’s Height vs Weight")

    # Scatter plot with magenta data points
    plt.scatter(x, y, color='magenta', alpha=0.5)

    # Display the plot
    plt.show()
