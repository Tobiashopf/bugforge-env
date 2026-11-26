"""Validate invoice and payment records in integer cents."""


def invoice_totals(invoices):
    if type(invoices) is not list:
        raise ValueError('invalid invoices')
    totals = {}
    for row in invoices:
        if type(row) not in (list, tuple) or len(row) != 2:
            raise ValueError('invalid invoice row')
        key, amount = row
        if type(key) is not str or not key.strip() or key in totals:
            raise ValueError('invalid or duplicate invoice id')
        if type(amount) is not int or amount <= 0:
            raise ValueError('invalid invoice amount')
        totals[key] = amount
    return totals


def unique_payments(payments):
    if type(payments) is not list:
        raise ValueError('invalid payments')
    unique = {}
    for row in payments:
        if type(row) not in (list, tuple) or len(row) != 3:
            raise ValueError('invalid payment row')
        key, invoice, cents = row
        if type(key) is not str or not key.strip() or type(invoice) is not str or not invoice.strip():
            raise ValueError('invalid payment identifiers')
        if type(cents) is not int or cents <= 0:
            raise ValueError('invalid payment amount')
        payload = (invoice, cents)
        if key in unique:
            if unique[key] != payload:
                raise ValueError('conflicting payment id')
            continue
        unique[key] = payload
    return unique
