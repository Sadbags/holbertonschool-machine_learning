#!/usr/bin/env python3
"""Module to add two 2D matrices of the same dimensions."""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element by element."""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
