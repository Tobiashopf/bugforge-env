"""Validate the two supported profile schemas."""


def identity(value):
    if type(value) is not str or not value.strip():
        raise ValueError('invalid identifier or name')


def contact(value):
    if value is not None and (type(value) is not str or not value.strip()):
        raise ValueError('invalid email')


def validate_v1(record):
    if (type(record) is not dict or not {'id', 'name'} <= set(record)
            or set(record) - {'id', 'name', 'email', 'active'}):
        raise ValueError('invalid v1 fields')
    identity(record['id'])
    identity(record['name'])
    contact(record.get('email'))
    if type(record.get('active', True)) is not bool:
        raise ValueError('invalid active flag')
    return True


def validate_v2(record):
    if type(record) is not dict or set(record) != {'id', 'profile', 'enabled'}:
        raise ValueError('invalid v2 fields')
    identity(record['id'])
    profile = record['profile']
    if type(profile) is not dict or set(profile) != {'name', 'email'}:
        raise ValueError('invalid profile fields')
    identity(profile['name'])
    contact(profile['email'])
    if type(record['enabled']) is not bool:
        raise ValueError('invalid enabled flag')
    return True
