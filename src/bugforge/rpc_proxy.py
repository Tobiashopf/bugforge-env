"""Trusted pytest-side proxy. This file is never mounted into the candidate container."""
from __future__ import annotations

import atexit
import json
import subprocess
from pathlib import Path

from _bf_codec import ERRORS, frame, pack, read_frame, unpack

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "_bf_config.json").read_text(encoding="utf-8"))
CLIENTS = {}


class ProtocolFailure(Exception):
    pass


class Client:
    def __init__(self, module):
        self.seq = 0
        self.failed = False
        self.proc = subprocess.Popen(CONFIG[module], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, cwd=ROOT.parent / "candidate")
        atexit.register(self.close)

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if self.proc.stdout:
            self.proc.stdout.close()

    def request(self, op, name, args, kwargs, handle=None):
        self.seq += 1
        try:
            if self.failed:
                raise ValueError("worker already invalid")
            message = {"id": self.seq, "op": op, "name": name, "args": pack(args),
                       "kwargs": pack(kwargs), "handle": handle}
            self.proc.stdin.write(frame(message))
            self.proc.stdin.flush()
            reply = read_frame(self.proc.stdout)
            if (type(reply) is not dict or type(reply.get("id")) is not int
                    or reply["id"] != self.seq):
                raise ValueError("invalid response identity")
            if set(reply) == {"id", "value"}:
                value = unpack(reply["value"])
                if op == "new" and value is not None:
                    raise ValueError("invalid constructor response")
                return self.seq if op == "new" else value
            if set(reply) != {"id", "error"} or reply["error"] not in ERRORS:
                raise ValueError("invalid response schema")
            error = ERRORS[reply["error"]]
        except (ValueError, TypeError, OSError, RecursionError, OverflowError) as exc:
            self.failed = True
            (ROOT / "_bf_protocol_failed").touch()
            raise ProtocolFailure("invalid candidate protocol") from exc
        # Domain exceptions are deliberately distinct from transport failures.
        raise error()


def client(module):
    if module not in CLIENTS:
        try:
            CLIENTS[module] = Client(module)
        except OSError as exc:
            (ROOT / "_bf_protocol_failed").touch()
            raise ProtocolFailure("candidate worker failed to start") from exc
    return CLIENTS[module]


def function(module, name):
    def call(*args, **kwargs):
        return client(module).request("call", name, args, kwargs)
    return call


def proxy_class(module, name, methods):
    class Proxy:
        def __init__(self, *args, **kwargs):
            self._client = client(module)
            self._handle = self._client.request("new", name, args, kwargs)

    def method(method_name):
        def call(self, *args, **kwargs):
            return self._client.request("method", method_name, args, kwargs, self._handle)
        return call

    for method_name in methods:
        setattr(Proxy, method_name, method(method_name))
    return Proxy
