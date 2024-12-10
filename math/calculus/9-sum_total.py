#!/usr/bin/env python3
"""This module calculates the sum of the squares """


def summation_i_squared(n):
    """This function calculates the sum of the squares"""
    # Validates input
    if not isinstance(n, int) or n <= 0:
        return None  # Returns none if n is not a valid number

    # Applies formula to calculate the sum of squares
    sum_squares = (n * (n + 1) * (2 * n + 1)) // 6

    # returns the result
    return sum_squares
