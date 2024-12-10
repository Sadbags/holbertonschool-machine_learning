#!/usr/bin/env python3
""" plot y as a line graph: """

import numpy as np
import matplotlib.pyplot as plt

def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, "r-")  # plot y as a red line
    plt.xlim([0, 10])  # sets x axis from 0

    plt.show() # this displays the plot
