import numpy as np
import sys

def sprint(text):
    sys.stdout.write(text)

def random_per_target(x, y, n=1, return_y=False):
    """
    take
    :param x:
    :param y:
    :param return_y:
    :return:
    """
    x = np.array(x)
    y = np.array(y)
    rx = []
    ry = []
    for i in np.unique(y):
        indices = np.where(y == i)[0]
        np.random.shuffle(indices)
        indices = indices[:n]
        rx.extend(x[indices])
        ry.extend(y[indices])
    rx = np.array(rx)
    ry = np.array(ry)
    if return_y:
        return rx, ry
    return rx