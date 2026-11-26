"""Sample Windows system counters during an episode. Values describe the whole host."""

import ctypes
import json
import os
import time
import uuid
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from threading import Event, Thread


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong)
        for name in (
            "total",
            "available",
            "page_total",
            "page_available",
            "virtual_total",
            "virtual_available",
            "extended_available",
        )
    ]


def sample():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.windll.kernel32
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    counters = [wintypes.FILETIME() for _ in range(3)]
    if not kernel.GetSystemTimes(*(ctypes.byref(c) for c in counters)):
        raise OSError("GetSystemTimes failed")
    idle, system, user = [(c.dwHighDateTime << 32) + c.dwLowDateTime for c in counters]
    return status.available, idle, system + user


@contextmanager
def monitor(directory: Path, labels):
    stop = Event()
    result = {**labels, "scope": "whole Windows host", "interval_seconds": 2, "samples": 0}
    started = time.monotonic()

    def collect():
        if os.name != "nt":
            result["unavailable"] = "Windows counters required"
            return
        previous = None
        try:
            while True:
                available, idle, total = sample()
                result["samples"] += 1
                result["min_available_bytes"] = min(
                    result.get("min_available_bytes", available), available
                )
                if previous and total > previous[1]:
                    busy = 100 * (1 - (idle - previous[0]) / (total - previous[1]))
                    result["peak_system_cpu_percent"] = max(
                        busy, result.get("peak_system_cpu_percent", 0)
                    )
                previous = idle, total
                if stop.wait(2):
                    return
        except OSError as exc:
            result["unavailable"] = str(exc)

    thread = Thread(target=collect, daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join()
        result["elapsed_seconds"] = time.monotonic() - started
        directory.mkdir(exist_ok=True, parents=True)
        (directory / f"{uuid.uuid4().hex}.json").write_text(json.dumps(result), encoding="utf-8")
