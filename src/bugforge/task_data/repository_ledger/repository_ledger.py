"""Replay a batch of transfers with transaction-ID idempotency."""
from ledger_rules import validate_transfer
from ledger_balances import apply_transfer


def replay(opening, transactions):
    balances = dict(opening)
    for account, amount in balances.items():
        if type(account) is not str or not account or type(amount) is not int or amount < 0:
            raise ValueError("invalid opening balance")
    seen = {}
    applied = []
    for tx_id, source, target, amount in transactions:
        validate_transfer(balances, tx_id, source, target, amount)
        payload = (source, target, amount)
        if tx_id in seen:
            if seen[tx_id] != payload:
                raise ValueError("conflicting transaction id")
            continue
        balances = apply_transfer(balances, source, target, amount)
        seen[tx_id] = payload
        applied.append(tx_id)
    return {"balances": balances, "applied": applied}
