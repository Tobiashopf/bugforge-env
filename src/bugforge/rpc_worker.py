"""Value-only protocol for candidate execution; no pickle or trusted test data."""
from __future__ import annotations

import contextlib
import importlib
import json
import math
import sys

MAX_FRAME = 262144
ERRORS = {cls.__name__: cls for cls in (ValueError, KeyError, TypeError,
          IndexError, ZeroDivisionError, OverflowError, RuntimeError, AttributeError)}


def pack(value, depth=0):
    if depth > 32:
        raise ValueError("value nested too deeply")
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind in (tuple, list):
        return {"t": kind.__name__, "v": [pack(v, depth + 1) for v in value]}
    if kind is dict:
        return {"t": "dict", "v": [[pack(k, depth + 1), pack(v, depth + 1)]
                                    for k, v in value.items()]}
    raise ValueError("unsupported value type")


def unpack(value, depth=0):
    if depth > 32:
        raise ValueError("value nested too deeply")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is not dict or set(value) != {"t", "v"} or type(value["v"]) is not list:
        raise ValueError("invalid value")
    tag, vals = value["t"], value["v"]
    if tag in ("tuple", "list"):
        result = [unpack(v, depth + 1) for v in vals]
        return tuple(result) if tag == "tuple" else result
    if tag == "dict":
        out = {}
        for pair in vals:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("invalid mapping")
            key = unpack(pair[0], depth + 1)
            if key in out:
                raise ValueError("duplicate key")
            out[key] = unpack(pair[1], depth + 1)
        return out
    raise ValueError("invalid tag")


def frame(value):
    data = json.dumps(value, allow_nan=False, ensure_ascii=True).encode("ascii") + b"\n"
    if len(data) > MAX_FRAME:
        raise ValueError("oversized frame")
    return data


def read_frame(stream):
    data = stream.readline(MAX_FRAME + 1)
    if not data or len(data) > MAX_FRAME or not data.endswith(b"\n"):
        raise ValueError("missing or oversized frame")
    return json.loads(data)


def main():
    # Only candidate code and this protocol helper live in the worker directory.
    wire = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module(sys.argv[1])
        objects = {}
        while True:
            try:
                request = read_frame(sys.stdin.buffer)
            except (ValueError, OSError):
                return
            seq = request["id"]
            try:
                op, name = request["op"], request["name"]
                args, kwargs = unpack(request["args"]), unpack(request["kwargs"])
                if op == "new":
                    objects[seq] = getattr(module, name)(*args, **kwargs)
                    result = None
                elif op == "method":
                    result = getattr(objects[request["handle"]], name)(*args, **kwargs)
                elif op == "call":
                    result = getattr(module, name)(*args, **kwargs)
                else:
                    raise RuntimeError("unknown operation")
            except Exception as exc:  # noqa: BLE001
                kind = type(exc).__name__
                response = {"id": seq, "error": kind if kind in ERRORS else "RuntimeError"}
            else:
                try:
                    response = {"id": seq, "value": pack(result)}
                except (ValueError, TypeError, RecursionError):
                    # Unsupported results must not masquerade as domain exceptions.
                    return
            wire.write(frame(response))
            wire.flush()


if __name__ == "__main__":
    main()
