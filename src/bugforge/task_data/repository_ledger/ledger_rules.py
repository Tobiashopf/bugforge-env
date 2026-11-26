"""Integer-cent transfer validation."""


def validate_transfer(balances, tx_id, source, target, amount):
    if type(tx_id) is not str or not tx_id:
        raise ValueError("invalid transaction id")
    if source not in balances or target not in balances or source == target:
        raise ValueError("invalid accounts")
    if type(amount) is not int or amount <= 0:
        raise ValueError("invalid amount")
