"""Apply a previously fitted standardizer without refitting."""
from numerical_stats import observed
import math


def transform(values, state):
    mean, scale = state['mean'], state['scale']
    if (type(mean) not in (int, float) or type(scale) not in (int, float)
            or not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0
            or abs(mean) > 1e12 or scale > 1e12 or scale < 1e-100):
        raise ValueError('invalid fitted state')
    output = []
    for value in values:
        x = observed(value)
        if x is None:
            x = mean
        output.append((x - mean) / scale)
    return output
