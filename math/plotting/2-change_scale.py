#!/usr/bin/env python3
"""Plot x ↦ y as a line graph."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Changes scale."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)

    plt.yscale('log')  # Sets y scale
    plt.xlim(0, 28650)  # Sets x axis range
    plt.xlabel("Time (years)")  # Label for x axis
    plt.ylabel("Fraction Remaining")  # Label for y axis
    plt.title("Exponential Decay of C-14")  # Sets title of plot

    plt.show()  # Displays the plot
