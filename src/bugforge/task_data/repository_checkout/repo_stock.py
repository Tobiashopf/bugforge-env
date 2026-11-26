"""Pure inventory reservation; no mutation of the caller's stock."""


def reserve(stock, demand):
    remaining = dict(stock)
    for sku, quantity in demand.items():
        if quantity <= 0 or remaining.get(sku, 0) < quantity:
            raise ValueError("insufficient stock")
        remaining[sku] -= quantity
    return remaining
