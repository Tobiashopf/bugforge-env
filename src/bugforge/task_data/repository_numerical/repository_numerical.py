"""Train/inference orchestration with a shared fitted state."""
from numerical_stats import fit as fit_stats
from numerical_transform import transform as apply_transform


def fit(training):
    return fit_stats(training)


def transform(values, state):
    return apply_transform(values, state)


def run(training, inference):
    state = fit(training)
    return {'state': state, 'training': transform(training, state),
            'inference': transform(inference, state)}
