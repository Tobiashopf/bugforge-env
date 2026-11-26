import pytest
from repository_api import validate_v1, validate_v2, upgrade, downgrade, convert


def v1():
    return {'id': 'u1', 'name': 'Ada', 'email': 'ada@example.org', 'active': False}


def v2():
    return {'id': 'u1', 'profile': {'name': 'Ada', 'email': 'ada@example.org'}, 'enabled': False}


def test_v1_unknown_fields():
    with pytest.raises(ValueError):
        validate_v1(dict(v1(), extra=7))


def test_v2_boolean():
    with pytest.raises(ValueError):
        validate_v2(dict(v2(), enabled=1))


def test_upgrade_default():
    assert upgrade({'id': 'u1', 'name': 'Ada'}) == {
        'id': 'u1', 'profile': {'name': 'Ada', 'email': None}, 'enabled': True}


def test_downgrade_email():
    assert downgrade(v2()) == v1()


def test_same_version_v2():
    assert convert(v2(), 2, 2) == v2()


def test_explicit_false():
    assert upgrade(v1()) == v2()


def test_roundtrip_v1():
    assert convert(convert(v1(), 1, 2), 2, 1) == v1()


def test_roundtrip_v2():
    assert convert(convert(v2(), 2, 1), 1, 2) == v2()


def test_canonical_v1():
    assert convert({'id': 'u2', 'name': 'Lin'}, 1, 1) == {
        'id': 'u2', 'name': 'Lin', 'email': None, 'active': True}


def test_invalid_versions():
    for source, target in [(True, 2), (1, False), (3, 2), (1, '2')]:
        with pytest.raises(ValueError):
            convert(v1(), source, target)


def test_invalid_fields():
    for record in [None, {'id': 'u'}, dict(v1(), name=' '), dict(v1(), email=5), dict(v1(), active=0)]:
        with pytest.raises(ValueError):
            upgrade(record)


def test_invalid_profile():
    for profile in [None, {'name': 'Ada'}, {'name': 'Ada', 'email': None, 'x': 1}]:
        with pytest.raises(ValueError):
            downgrade(dict(v2(), profile=profile))
