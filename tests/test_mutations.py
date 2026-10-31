import ast

import pytest

from bugforge import base
from bugforge.mutations import OPERATORS, apply, enumerate_sites, normalize


@pytest.mark.parametrize("name", base.available())
def test_every_site_produces_valid_different_code(name):
    src = base.load(name).source
    for site in enumerate_sites(src):
        mutant = apply(src, [site.id])
        ast.parse(mutant)
        assert mutant != src, site


@pytest.mark.parametrize("name", base.available())
def test_site_enumeration_is_deterministic(name):
    src = base.load(name).source
    assert enumerate_sites(src) == enumerate_sites(src)
    assert enumerate_sites(src) == enumerate_sites(normalize(src))


def test_all_operators_are_exercised_somewhere():
    seen = {s.operator for n in base.available() for s in enumerate_sites(base.load(n).source)}
    assert seen == set(OPERATORS)


def test_multiple_sites_apply_together():
    src = base.load("pagination").source
    sites = [s.id for s in enumerate_sites(src) if s.operator == "cmp"][:2]
    both = apply(src, sites)
    assert both != apply(src, sites[:1]) != src


def test_unknown_site_raises():
    with pytest.raises(KeyError):
        apply("def f(x):\n    return x\n", ["cmp:999"])


def test_module_level_code_is_not_mutated():
    src = 'LIMIT = 5\n\ndef f(x):\n    return x < 3\n'
    sites = enumerate_sites(src)
    assert [s.operator for s in sites] == ["cmp", "const"]


def test_normalize_is_idempotent():
    src = base.load("inventory").source
    assert normalize(src) == src
