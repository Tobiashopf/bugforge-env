"""Version-aware profile API with canonical same-version responses."""
from api_schema import validate_v1 as check_v1, validate_v2 as check_v2
from api_migration import upgrade as to_v2, downgrade as to_v1


def validate_v1(record):
    return check_v1(record)


def validate_v2(record):
    return check_v2(record)


def upgrade(record):
    return to_v2(record)


def downgrade(record):
    return to_v1(record)


def convert(record, source, target):
    if type(source) is not int or type(target) is not int or source not in (1, 2) or target not in (1, 2):
        raise ValueError('unsupported version')
    if source == 1:
        canonical = upgrade(record)
    else:
        validate_v2(record)
        canonical = {'id': record['id'], 'profile': dict(record['profile']), 'enabled': record['enabled']}
    if target == 1:
        return downgrade(canonical)
    return canonical
