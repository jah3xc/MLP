import numpy as np
import math
from constants import *


def calc_v(inputs, weights, bias):
    """
    Calculate v, the input vector
    v = wTp + b
    """
    return np.dot(inputs, weights) + bias


def fi_prime(v):
    """
    Return the value of the activation function's derivative
    """

    # calc fi
    f = fi(v)
    # fi * 1 - fi
    return f * (1 - f)


def adjust_b(bias, delta):
    """
    Adjust the bias
    bias += ALPHA * 1 * delta
    """
    return bias + (ALPHA * delta)


def fi(v):
    """
    Define the activation function and return fi of v
    """

    # define the sigmoid and return the value
    denom = 1 + math.e ** (-1 * v)
    val = 1 / denom
    return val


def calc_error(output, desired_output):
    """
    Calculate the error at the output layer
    """
    er = 0
    # iterate throuh all output neurons
    for y, d in zip(output, desired_output):
        # add this error to total error
        er += (y - d)**2

    return er
