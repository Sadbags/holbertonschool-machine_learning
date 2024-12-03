#!/usr/bin/env python3
"""module to calculate the shape of a matrix"""
def matrix_shape(matrix):
    """calculates the shape of a matrix, and must return a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
