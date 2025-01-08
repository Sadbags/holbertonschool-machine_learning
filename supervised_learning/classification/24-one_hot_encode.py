#!/usr/bin/env python3
""" This module create a one hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """ This function create a one hot matix from a numpy array """
    # Check if Y is a numpy array and classes is an integer
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    # Check if Y is empty
    if len(Y) == 0:
        return None
    # Check if classes <= max value in Y
    if classes <= np.max(Y):
        return None
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
