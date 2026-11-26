"""Translate validated records between versioned contracts."""
from api_schema import validate_v1, validate_v2


def upgrade(record):
    validate_v1(record)
    return {'id': record['id'], 'profile': {'name': record['name'],
            'email': record.get('email')}, 'enabled': record.get('active', True)}


def downgrade(record):
    validate_v2(record)
    return {'id': record['id'], 'name': record['profile']['name'],
            'email': record['profile']['email'], 'active': record['enabled']}
