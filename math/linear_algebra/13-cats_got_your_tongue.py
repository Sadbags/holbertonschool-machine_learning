#!/usr/bin/env python3
""" module that concatenates two matrices along a specific axis """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    if axis == 0:
        return np.concatenate((mat1, mat2), axis=axis)
    elif axis == 1:
        return np.concatenate((mat1, mat2), axis=axis)
    else:
        return None
