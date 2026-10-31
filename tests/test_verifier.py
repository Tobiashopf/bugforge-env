import pytest

from bugforge import base, policy, verifier
from bugforge.verifier import verify

INV = base.load("inventory")


def test_original_is_solved():
    res = verify({INV.module_file: INV.source}, [INV])
    assert res.solved and res.passed == res.expected


def test_bugged_module_is_not_solved():
    bugged = INV.source.replace("if qty <= 0:", "if qty < 0:", 1)
    res = verify({INV.module_file: bugged}, [INV])
    assert not res.solved and res.failed


def test_missing_file_is_a_violation():
    res = verify({}, [INV])
    assert not res.solved and INV.module_file in res.violations


def test_infinite_loop_times_out():
    src = INV.source.replace("def quantity(self, sku):",
                             "def quantity(self, sku):\n        while True:\n            pass", 1)
    res = verify({INV.module_file: src}, [INV], timeout=5)
    assert res.timed_out and not res.solved


def test_fuzz_catches_hardcoding_the_visible_answers():
    # Passes every unit test by special-casing their inputs, but is wrong elsewhere.
    src = INV.source.replace(
        "if subtotal >= discount_threshold:",
        "if subtotal >= discount_threshold and subtotal in (100.0, 120.0, 60.0):", 1)
    res = verify({INV.module_file: src}, [INV])
    assert not res.solved
    assert any("fuzz" in f for f in res.failed)


# Defense in depth: with the static policy switched off, the later layers must
# still refuse to pay out for these.

@pytest.fixture
def no_policy(monkeypatch):
    monkeypatch.setattr(policy, "check", lambda src: [])
    monkeypatch.setattr(verifier.policy, "check", lambda src: [])


def test_exit_at_import_without_policy_still_fails(no_policy):
    src = "import os\nos._exit(0)\n" + INV.source
    res = verify({INV.module_file: src}, [INV])
    assert not res.solved and res.total == 0


def test_module_level_skip_without_policy_still_fails(no_policy):
    src = "import pytest\npytest.skip('bye', allow_module_level=True)\n" + INV.source
    res = verify({INV.module_file: src}, [INV])
    assert not res.solved
