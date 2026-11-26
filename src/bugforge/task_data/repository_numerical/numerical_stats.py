"""Fit a one-dimensional standardizer using observed training values."""
import math


def observed(value):
    if value is None:
        return None
    if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
        raise ValueError('invalid observation')
    return float(value)


def fit(training):
    values = [observed(x) for x in training if x is not None]
    if not values:
        raise ValueError('no observed training values')
    mean = math.fsum(values) / len(values)
    squares = [(x - mean) ** 2 for x in values]
    variance = math.fsum(squares) / len(squares)
    scale = math.sqrt(variance) if variance >= 1e-200 else 1.0
    return {'mean': mean, 'scale': scale}
