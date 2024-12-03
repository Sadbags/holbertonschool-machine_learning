#!/usr/bin/env python3
"""module to flip a matrix"""


def matrix_transpose(matrix):
    """flips a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
