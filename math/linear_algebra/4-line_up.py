#!/usr/bin/env python3
"""module to add two arrays"""


def add_arrays(arr1, arr2):
    """add two arrays"""
    if len(arr1) != len(arr2):
        return none
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
