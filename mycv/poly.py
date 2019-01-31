from typing import Callable, Any
from collections import namedtuple

import numpy as np


# Fit is a class containing the parameters of a polynomial fit expressed as
# a.x^2 + b.x + c
Fit = namedtuple('Fit', ['a', 'b', 'c'])


def zero() -> Fit:
    '''zero returns the polynomial equal to 0 everywhere'''
    return Fit(a=0, b=0, c=0)


def from_numpy(fit:np.array) -> Fit:
    '''from_numpy converts the result of np.polyfit() to a Fit'''
    return Fit(a=fit[0], b=fit[1], c=fit[2])


def fit(x:np.array, y:np.array) -> Fit:
    '''fit fits a quadratic polynomial to the given data'''
    return from_numpy(np.polyfit(x, y, 2))


def compute(poly:Fit, x:np.array) -> np.array:
    '''compute computes the y-values of the polynomial given the x-values'''
    return poly.a * np.square(x) + poly.b * x + poly.c


def in_window(fit:Fit, x:np.array, y:np.array, margin:int):
    '''
    returns a mask of the x-array set to 1 where x is within the window
    computed from the polynomial
    '''
    window = compute(fit, y)
    mask = (x >= window - margin) & (x < window + margin)
    return mask

def curve(fit:Fit) -> Callable[[np.array], np.array]:
    '''
    curve returns a function that computes the radius of curvature of a
    polynomial as a function of x
    '''
    return lambda x: np.sqrt(((1 + (2 * fit.a * x + fit.B)**2)**3)/(4 * fit.A**2))