from typing import Callable, Any, List
from collections import namedtuple

import numpy as np


# Fit is a class containing the parameters of a polynomial fit expressed as
# a.x^2 + b.x + c
Fit = namedtuple('Fit', ['a', 'b', 'c'])


def constant(c:float) -> Fit:
    '''zero returns the polynomial equal to 0 everywhere'''
    return Fit(a=0, b=0, c=c)


def from_numpy(fit:np.array) -> Fit:
    '''from_numpy converts the result of np.polyfit() to a Fit'''
    return Fit(a=fit[0], b=fit[1], c=fit[2])


def to_numpy(fit:Fit) -> np.array:
    return np.array([fit.a, fit.b, fit.c])


def fit(x:np.array, y:np.array) -> Fit:
    '''fit fits a quadratic polynomial to the given data'''
    return from_numpy(np.polyfit(x, y, 2))


def compute(fit:Fit, x:np.array) -> np.array:
    '''compute computes the y-values of the polynomial given the x-values'''
    return fit.a * np.square(x) + fit.b * x + fit.c


def mix(a:float, b:float, k:float) -> float:
    '''k must be between 0 and 1'''
    return a * k + b * (1-k)


def mix_curvatures(this:Fit, other:Fit, c:float) -> np.array:
    '''c must be between 0 and 1'''
    return Fit(a=mix(this.a, other.a, c), b=mix(this.b, other.b, c), c=this.c)


def in_window(fit:Fit, x:np.array, y:np.array, margin:int):
    '''
    returns a mask of the x-array set to 1 where x is within the window
    computed from the polynomial
    '''
    window = compute(fit, y)
    mask = (x >= window - margin) & (x < window + margin)
    return mask


def curvature(fit:Fit) -> Callable[[np.array], np.array]:
    '''
    curvature returns a function that computes the radius of curvature of a
    polynomial as a function of x
    '''
    if fit.a == 0:
        return lambda x: float('Inf')
    return lambda x: np.sqrt(((1 + (2 * fit.a * x + fit.b)**2)**3)/(4 * fit.a**2))


def average(fits:List[Fit]) -> Fit:
    return from_numpy(np.mean(np.array([to_numpy(fit) for fit in fits]), axis=0))


def convert_units(fit:Fit, cx:float, cy:float) -> Fit:
    a = fit.a*cy/cx**2
    b = fit.b*cy/cx
    c = fit.c*cy
    return Fit(a=a, b=b, c=c)