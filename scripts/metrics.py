"""
Helper functions for calculating signal metrics.
"""

import numpy as np

def normalize(x, axis=-1):
    """Normalize x by subtracting the mean and dividing by the st. dev."""
    if axis is None or x.ndim == 1:
        ### normalize across all axes
        return (x - x.mean()) / x.std()
    else:
        ### normalize across the given axis
        b = [slice(None) for i in xrange(x.ndim)]
        b[axis] = None
        return (x - x.mean(axis)[b]) / x.std(axis)[b]

def rms(x, axis=-1):
    return np.sqrt((x**2).mean(axis=axis))

def rmse(x, y, axis=-1):
    return rms(x - y, axis=axis)

def rmse_n(x, y, axis=-1):
    """Root-mean-square error normalized by the input signal RMS amplitude"""
    return rms(x - y, axis=axis) / rms(x, axis=axis)

def mutual_info(x, y, bins=19):
    x = x.flatten()
    y = y.flatten()
    assert len(x) == len(y)
    n = float(len(x))

    Pab, xedges, yedges = np.histogram2d(x, y, bins=bins)
    Pab /= n
    Pa = Pab.sum(axis=1)
    Pb = Pab.sum(axis=0)

    PaPb = np.outer(Pa, Pb)
    m = Pab > 1/n
    mutualinfo = (Pab[m] * np.log2(Pab[m] / PaPb[m])).sum()
    return mutualinfo
