#!/usr/bin/env python3
"""Plot all 5 previous graphs in one figure."""

import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """Plot all 5 previous graphs in one figure."""

    # Generate y values for cubic function (y0 = x^3)
    y0 = np.arange(0, 11) ** 3

    # Define mean and covariance for 2D normal distribution
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)  # Set random seed for reproducibility
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T  # Generate random data points
    y1 += 180  # Adjust y1 values by adding 180

    # Define x values and calculate exponential decay
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)  # Exponential decay for C-14

    # Define second exponential decay with different time constant
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)  # Exponential decay for C-14
    y32 = np.exp((r3 / t32) * x3)  # Exponential decay for Ra-226

    # Generate random student grades from normal distribution
    np.random.seed(5)  # Set random seed for reproducibility
    student_grades = np.random.normal(68, 15, 50)  # 50 random grades with mean 68 and std 15

    # Create a figure for the subplots
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle("All in One")  # Set the title for the entire figure

    # Plot 1: Cubic function
    plt.subplot(3, 2, 1)  # Create the first subplot in a 3x2 grid
    plt.plot(y0, "r-")  # Plot y0 as a red line
    plt.xticks(range(0, 11, 2))  # Set x ticks from 0 to 10 with step 2
    plt.yticks(range(0, 1100, 500))  # Set y ticks from 0 to 1000 with step 500
    plt.xlim(0, 10)  # Set x-axis limits from 0 to 10

    # Plot 2: Scatter plot of height vs weight
    plt.subplot(322)  # Create the second subplot
    plt.scatter(x1, y1, c='magenta')  # Scatter plot with magenta points
    plt.xlabel('Height (in)', fontsize='x-small')  # Label x-axis
    plt.ylabel('Weight (lbs)', fontsize='x-small')  # Label y-axis
    plt.title('Men\'s Height vs Weight', fontsize='x-small')  # Title of the plot

    # Plot 3: Exponential decay of C-14
    plt.subplot(3, 2, 3)  # Create the third subplot
    plt.plot(x2, y2)  # Plot exponential decay of C-14
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xticks(range(0, 30000, 10000))  # Set x ticks from 0 to 30000 with step 10000
    plt.xlim(0, 28650)  # Set x-axis limits
    plt.xlabel('Time (years)', fontsize='x-small')  # Label x-axis
    plt.ylabel('Fraction Remaining', fontsize='x-small')  # Label y-axis
    plt.title('Exponential Decay of C-14', fontsize='x-small')  # Title of the plot

    # Plot 4: Exponential decay of two elements (C-14 and Ra-226)
    plt.subplot(324)  # Create the fourth subplot
    plt.plot(x3, y31, 'r--', label='C-14')  # Plot C-14 decay with red dashed line
    plt.plot(x3, y32, 'g-', label='Ra-226')  # Plot Ra-226 decay with green solid line
    plt.xticks(np.arange(0, 21000, step=5000))  # Set x ticks from 0 to 20000 with step 5000
    plt.xlabel('Time (years)', fontsize='x-small')  # Label x-axis
    plt.xlim(0, 20000)  # Set x-axis limits
    plt.yticks(np.arange(0, 1.2, step=0.5))  # Set y ticks from 0 to 1 with step 0.5
    plt.ylabel('Fraction Remaining', fontsize='x-small')  # Label y-axis
    plt.ylim(0, 1)  # Set y-axis limits
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')  # Title of the plot
    plt.legend(fontsize='x-small')  # Display legend

    # Plot 5: Histogram of student grades
    plt.subplot(3, 2, (5, 6))  # Create the fifth subplot spanning columns 5 and 6
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')  # Plot histogram with black outlines
    plt.xticks(range(0, 101, 10))  # Set x ticks from 0 to 100 with step 10
    plt.yticks(range(0, 31, 10))  # Set y ticks from 0 to 30 with step 10
    plt.xlabel('Grades', fontsize='x-small')  # Label x-axis
    plt.ylabel('Number of Students', fontsize='x-small')  # Label y-axis
    plt.title('Project A', fontsize='x-small')  # Title of the plot
    plt.xlim(0, 100)  # Set x-axis limits
    plt.ylim(0, 30)  # Set y-axis limits

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot

