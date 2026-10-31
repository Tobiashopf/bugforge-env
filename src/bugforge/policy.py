"""Static policy for submitted modules: the first line of defense against reward hacking.

The verifier runs the agent's code in the same process as the hidden tests, so
code that runs at import time could rewrite the test framework, exit the
process with status 0, or reach the reference implementation. The fixes this
environment asks for never need any of that, so the policy forbids it outright:

- top level may only hold imports, definitions, assignments and a docstring
  (no statements that run code on import);
- imports are restricted to a small allow-list of pure standard-library modules;
- no function calls outside function bodies (class bodies, decorators and
  default arguments run at import time too);
- no dynamic-execution or introspection builtins, and no dunder attributes.

A legitimate fix is never blocked by this. Anything it blocks scores zero.
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"math", "collections", "dataclasses", "typing", "functools",
                   "itertools", "re", "string", "__future__"}
FORBIDDEN_NAMES = {"exec", "eval", "compile", "open", "__import__", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "exit",
                   "quit", "__builtins__", "memoryview", "help"}
_TOP_LEVEL_OK = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.ClassDef, ast.Assign, ast.AnnAssign)


def check(source: str) -> list[str]:
    """Return a list of violations; empty means the module is acceptable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]

    problems: list[str] = []
    for i, stmt in enumerate(tree.body):
        is_docstring = (i == 0 and isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str))
        if not (is_docstring or isinstance(stmt, _TOP_LEVEL_OK)):
            problems.append(f"line {stmt.lineno}: top-level {type(stmt).__name__} not allowed")

    in_body: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            body = node.body if isinstance(node.body, list) else [node.body]
            for stmt in body:
                in_body.update(id(sub) for sub in ast.walk(stmt))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and id(node) not in in_body:
            problems.append(f"line {node.lineno}: call outside a function body not allowed")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of '{alias.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: import from '{node.module}' not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: use of '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"line {node.lineno}: dunder attribute '{node.attr}' not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: {type(node).__name__.lower()} not allowed")
    return problems
