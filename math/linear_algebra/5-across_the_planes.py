#!/usr/bin/env python3
"""module to concatenate two matrices along a specific axis"""

def add_matrices2D(mat1, mat2):
    """adds two matrices"""
    if len(mat1) != len(mat2):
        return None
    return [add_arrays(mat1[i], mat2[i]) for i in range(len(mat1))]
