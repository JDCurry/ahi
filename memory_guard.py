"""
Give memory back to the operating system when the app goes quiet.

Why this exists. Render's metrics showed the service pinned at 1.9 GB of its
2 GB limit for days with almost no traffic, then killed and restarted (the 502).
Two things kept memory high between visits:

  1. Streamlit only enforces a cache's ttl when that cache is next read. On a
     quiet site nothing reads it, so a state loaded on Tuesday is still held on
     Friday.
  2. Freeing Python objects does not return the memory to the OS. glibc keeps
     freed pages in its heap, so the process size never comes back down. Only
     malloc_trim() forces it.

What it does. A single background thread per process. Every page interaction
calls touch(). If nothing has touched the app for `idle_seconds`, the thread
clears the heavy caches it was given, runs the garbage collector, releases
Arrow's memory pool, and calls malloc_trim(0). It does this once per quiet
period, not on every tick, and logs the process size before and after so the
effect is visible in Render's logs.

What it deliberately does not do. It never runs while someone is using the app,
and it leaves the loaded models alone: they are a fixed cost, reloading them
would make the next visitor wait, and they are not what grows.

Closing a browser tab is not a usable signal. Streamlit caches are shared by
the whole process rather than owned by one visitor, so a tab closing frees
nothing on its own; a quiet period is the reliable trigger.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time
from typing import Callable, Iterable

_last_activity = time.monotonic()
_trimmed_since_activity = False
_lock = threading.Lock()
_started = False


def touch() -> None:
    """Record activity. Call at the top of every script run."""
    global _last_activity, _trimmed_since_activity
    _last_activity = time.monotonic()
    _trimmed_since_activity = False


def rss_mb() -> float | None:
    """Current resident set size in MB, where the OS exposes it cheaply."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def release(clear: Iterable[Callable[[], None]] = ()) -> None:
    """Drop the given caches and hand freed memory back to the OS."""
    for fn in clear:
        try:
            fn()
        except Exception as exc:  # a failed clear must never take the thread down
            print(f'[memory_guard] clear failed: {exc!r}', file=sys.stderr, flush=True)
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)  # glibc only, which is what Render runs
    except (OSError, AttributeError):
        pass  # Windows or macOS during development: gc.collect() is all there is


def start(clear: Iterable[Callable[[], None]], idle_seconds: int = 600,
          check_every: int = 60) -> None:
    """Start the idle sweeper once per process. Safe to call on every rerun."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    callbacks = list(clear)

    def loop() -> None:
        global _trimmed_since_activity
        while True:
            time.sleep(check_every)
            idle = time.monotonic() - _last_activity
            if idle < idle_seconds or _trimmed_since_activity:
                continue
            before = rss_mb()
            release(callbacks)
            _trimmed_since_activity = True
            after = rss_mb()
            if before is not None and after is not None:
                print(f'[memory_guard] idle {idle / 60:.0f} min: released '
                      f'{before - after:,.0f} MB ({before:,.0f} -> {after:,.0f} MB)',
                      file=sys.stderr, flush=True)

    threading.Thread(target=loop, name='memory_guard', daemon=True).start()
