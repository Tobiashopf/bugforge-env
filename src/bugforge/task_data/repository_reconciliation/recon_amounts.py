"""Allocate a payment to available invoice balance, leaving excess unapplied."""


def allocate(amount, remaining):
    applied = min(amount, remaining)
    return applied, amount - applied
