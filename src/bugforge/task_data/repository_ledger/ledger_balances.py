"""Pure transfer on a private balance snapshot."""


def apply_transfer(balances, source, target, amount):
    if balances[source] < amount:
        raise ValueError("insufficient funds")
    updated = dict(balances)
    updated[source] -= amount
    updated[target] += amount
    return updated
