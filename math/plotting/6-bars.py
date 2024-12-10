#!/usr/bin/env python3
"""Plot a stacked bar graph."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))  # Generate random fruit data
    plt.figure(figsize=(6.4, 4.8))  # Set figure size

    names = ['Farrah', 'Fred', 'Felicia']  # Names of people
    fruits = [('apples', 'red'), ('bananas', 'yellow'),
              ('oranges', '#ff8000'), ('peaches', '#ffe5b4')]
    bottom = np.zeros(len(names))  # Initialize bottom for stacking

    # Plot stacked bars for each fruit
    for fruit_name, color in fruits:
        plt.bar(names, fruit[fruits.index((fruit_name, color))],
                bottom=bottom, color=color, width=0.5)
        bottom += fruit[fruits.index((fruit_name, color))]

    # Add legend for each fruit
    plt.legend([fruit_name for fruit_name, color in fruits])

    # Label for y-axis and set ticks
    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))

    # Set title of the plot
    plt.title('Number of Fruit per Person')

    plt.show()  # Display the plot
