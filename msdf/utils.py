from time import time

from numpy.core._multiarray_umath import sqrt


def millis():
    return int(time() * 1000)


def dist(x, y):
    x = x.flatten()
    y = y.flatten()
    return sqrt((y[1] - x[1])**2 + (y[0] - x[0])**2)