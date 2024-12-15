#!/usr/bin/env python3
""" This module contains a
function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ Calculate the integral of a polynomial """
    # Check if poly is a list and is not empty, and if C is an integer
    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None

    integral = [C]

    # For each coefficient in the polynomial...
    for i in range(len(poly)):
        coef = poly[i] / (i + 1)

        # If the coefficient is a whole number, represent it as an integer
        if coef.is_integer():
            coef = int(coef)

        # Append the coefficient to the list of integral coefficients
        integral.append(coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
