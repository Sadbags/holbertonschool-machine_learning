#!/usr/bin/env python3
import numpy as np
matrix = np.array([[36, 14, 57, 82, -9, 10],
                   [100, 109, -36, 7, 2, 443],
                   [-6, 54, 57, 82, -36, 7],
                   [23, 72, 72, 12, 21, 8]])

print("The middle two rows of the matrix are:\n{}".format(matrix[1:3]))
print("The middle two columns of the matrix are:\n{}".format(matrix[:, 2:4]))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(matrix[1:, 3:]))
