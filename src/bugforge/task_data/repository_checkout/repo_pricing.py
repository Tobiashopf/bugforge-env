"""Integer-cent pricing contract."""


def line_total(unit_cents, quantity):
    if unit_cents < 0 or quantity <= 0:
        raise ValueError("invalid line")
    return unit_cents * quantity


def discount(subtotal, member):
    if member and subtotal >= 1000:
        return subtotal // 10
    return 0
