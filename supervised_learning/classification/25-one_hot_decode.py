#!/usr/bin/env python3
""" this module has a func that converts a one hot matrix """

import numpy as np


def one_hot_decode(one_hot):
    """ converts a one shot matric into a vector """
    if not isinstance(one_hot, np.ndarray) or len(
            one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
