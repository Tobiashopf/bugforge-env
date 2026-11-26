"""Reconcile a finite batch without external state or floating-point money."""
from recon_rules import invoice_totals, unique_payments
from recon_amounts import allocate


def reconcile(invoices, payments):
    outstanding = invoice_totals(invoices)
    unique = unique_payments(payments)
    allocations = {}
    for key, (invoice, amount) in unique.items():
        applied, unapplied = allocate(amount, outstanding.get(invoice, 0))
        if invoice in outstanding:
            outstanding[invoice] -= applied
        allocations[key] = {'invoice': invoice, 'applied': applied, 'unapplied': unapplied}
    return {'outstanding': outstanding, 'payments': allocations}
