import pytest

from bugforge import base, policy


@pytest.mark.parametrize("name", base.available())
def test_base_modules_pass_policy(name):
    assert policy.check(base.load(name).source) == []


def test_typical_fix_passes():
    src = "import math\n\ndef area(r):\n    return math.pi * r ** 2\n"
    assert policy.check(src) == []


@pytest.mark.parametrize("src", [
    "import os\n",
    "import sys\n",
    "import _pytest.outcomes\n",
    "from ref_inventory import *\n",
    "from . import x\n",
    "print('hi')\n",
    "x = open('f')\n",
    "def f():\n    return eval('1')\n",
    "def f():\n    return getattr(1, 'real')\n",
    "def f():\n    return ().__class__\n",
    "class A:\n    x = len([1])\n",
    "def f(x=len([1])):\n    return x\n",
    "def f():\n    global y\n",
    "def f(:\n",
])
def test_dangerous_code_is_rejected(src):
    assert policy.check(src) != []
