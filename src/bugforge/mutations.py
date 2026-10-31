"""AST mutation operators: the bugs the environment injects.

A *site* is one place in a module where one operator can apply one change. Sites
are enumerated in a deterministic order (``ast.walk`` over the parsed source),
so a site id like ``"cmp:57"`` names the same change every time the same source
is parsed. That is what lets the kill-matrix index be computed once and reused.

Only code inside function bodies is mutated: module docstrings and top-level
constants are left alone.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

_CMP_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_ARITH_FLIP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_BOOL_FLIP = {ast.And: ast.Or, ast.Or: ast.And}

OPERATORS = {
    "cmp": "flip a comparison boundary (< <=, > >=, == !=)",
    "arith": "swap an arithmetic operator (+ -, * /), including augmented assignment",
    "const": "off-by-one on an integer or float literal",
    "bool": "swap and/or",
    "negate": "negate an if-condition",
    "noraise": "remove a raise (a missing guard)",
    "reverse": "flip a sort direction (reverse=True/False)",
}


@dataclass(frozen=True)
class Site:
    id: str            # "<operator>:<walk index>"
    operator: str
    line: int
    description: str


def _function_nodes(tree: ast.Module) -> set[int]:
    """ids of every node that lives inside a function body."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    inside.add(id(sub))
    return inside


def _is_docstring(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    return isinstance(parent, ast.Expr) and isinstance(node, ast.Constant) and isinstance(
        node.value, str)


def _candidates(tree: ast.Module):
    """Yield (site, node) in deterministic order."""
    inside = _function_nodes(tree)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for idx, node in enumerate(ast.walk(tree)):
        if id(node) not in inside:
            continue
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_FLIP:
            new = _CMP_FLIP[type(node.ops[0])].__name__
            yield Site(f"cmp:{idx}", "cmp", line,
                       f"{type(node.ops[0]).__name__} -> {new}"), node
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _ARITH_FLIP:
            new = _ARITH_FLIP[type(node.op)].__name__
            yield Site(f"arith:{idx}", "arith", line,
                       f"{type(node.op).__name__} -> {new}"), node
        elif (isinstance(node, ast.Constant) and type(node.value) in (int, float)
              and not _is_docstring(node, parents)):
            yield Site(f"const:{idx}", "const", line, f"{node.value!r} -> {node.value + 1!r}"), node
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_FLIP:
            yield Site(f"bool:{idx}", "bool", line, f"{type(node.op).__name__} swapped"), node
        elif isinstance(node, ast.If):
            yield Site(f"negate:{idx}", "negate", line, "if-condition negated"), node
        elif isinstance(node, ast.Raise):
            yield Site(f"noraise:{idx}", "noraise", line, "raise removed"), node
        elif (isinstance(node, ast.keyword) and node.arg == "reverse"
              and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)):
            yield Site(f"reverse:{idx}", "reverse", getattr(node.value, "lineno", 0),
                       "sort direction flipped"), node


def enumerate_sites(source: str) -> list[Site]:
    """Every mutation site in a module, in deterministic order."""
    return [site for site, _ in _candidates(ast.parse(source))]


class _Apply(ast.NodeTransformer):
    def __init__(self, actions: dict[int, str]):
        self.actions = actions

    def generic_visit(self, node):
        super().generic_visit(node)
        op = self.actions.get(id(node))
        if op is None:
            return node
        if op == "cmp":
            node.ops = [_CMP_FLIP[type(node.ops[0])]()]
        elif op == "arith":
            node.op = _ARITH_FLIP[type(node.op)]()
        elif op == "const":
            node.value = node.value + 1
        elif op == "bool":
            node.op = _BOOL_FLIP[type(node.op)]()
        elif op == "negate":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif op == "noraise":
            return ast.Pass()
        elif op == "reverse":
            node.value = ast.Constant(value=not node.value.value)
        return node


def apply(source: str, site_ids: list[str]) -> str:
    """Return the source with every listed site mutated at once."""
    tree = ast.parse(source)
    by_id = {site.id: node for site, node in _candidates(tree)}
    missing = [s for s in site_ids if s not in by_id]
    if missing:
        raise KeyError(f"unknown mutation sites: {missing}")
    actions = {id(by_id[s]): s.split(":")[0] for s in site_ids}
    tree = ast.fix_missing_locations(_Apply(actions).visit(tree))
    return ast.unparse(tree) + "\n"


def normalize(source: str) -> str:
    """Round-trip through the AST so originals and mutants share one formatting.

    Without this, a mutant (re-printed by ``ast.unparse``) would differ from its
    original in whitespace and comments everywhere, and a diff would point
    straight at nothing useful -- or worse, the formatting itself would leak
    which files were touched.
    """
    return ast.unparse(ast.parse(source)) + "\n"
