"""Checkout service orchestrating pricing and stock reservation."""
import repo_pricing
from repo_stock import reserve


def checkout(lines, stock, member=False):
    demand = {}
    subtotal = 0
    for sku, unit_cents, quantity in lines:
        subtotal += repo_pricing.line_total(unit_cents, quantity)
        demand[sku] = demand.get(sku, 0) + quantity
    remaining = reserve(stock, demand)
    saved = repo_pricing.discount(subtotal, member)
    return {"subtotal": subtotal, "discount": saved,
            "total": subtotal - saved, "remaining": remaining}
