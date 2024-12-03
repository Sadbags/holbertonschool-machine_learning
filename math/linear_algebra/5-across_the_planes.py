#!/usr/bin/env python3
"""module to concatenate two matrices along a specific axis"""


def add_matrices2D(mat1, mat2):
    """adds two matrices"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
			for i in range(len(mat1))]
    if axis == 0:
        return [add_matrices2D(row1, row2) for row1, row2 in zip(mat1, mat2)]
    else:
        return [[add_matrices2D([mat1[i][j] for i in range(len(mat1))], mat2[j][i]) for j in range(len(mat2[0]))]
				for i in range(len(mat2))]

