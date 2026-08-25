# ActivitySim
# See full license in LICENSE.txt.
"""Tests for the cgroup-aware memory-ceiling helpers used by adaptive chunking (chunk_size_mode=auto)."""
from __future__ import annotations

import os

import psutil

from activitysim.core import mem

GIB = 1024**3


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_finite_limit_parsing():
    assert mem._finite_limit("62000000000") == 62000000000
    assert mem._finite_limit("max") is None
    assert mem._finite_limit(None) is None
    assert mem._finite_limit("garbage") is None
    assert mem._finite_limit("0") is None  # non-positive is not a real limit
    assert mem._finite_limit(str(mem._CGROUP_UNLIMITED)) is None  # unlimited sentinel


def test_memory_limit_cgroup_v2(tmp_path):
    root = str(tmp_path)
    _write(root, "memory.max", "60000000000\n")
    assert mem.get_memory_limit(cgroup_root=root) == 60000000000


def test_memory_limit_cgroup_v2_max_falls_back(tmp_path):
    # cgroup v2 present but unlimited ("max") -> fall through to host RAM (a positive int)
    root = str(tmp_path)
    _write(root, "memory.max", "max\n")
    limit = mem.get_memory_limit(cgroup_root=root)
    assert limit == int(psutil.virtual_memory().total)
    assert limit > 0


def test_memory_limit_cgroup_v1(tmp_path):
    root = str(tmp_path)  # no memory.max -> v1 path
    _write(root, "memory/memory.limit_in_bytes", "48000000000\n")
    assert mem.get_memory_limit(cgroup_root=root) == 48000000000


def test_memory_limit_cgroup_v1_unlimited_falls_back(tmp_path):
    root = str(tmp_path)
    _write(root, "memory/memory.limit_in_bytes", str(mem._CGROUP_UNLIMITED))
    assert mem.get_memory_limit(cgroup_root=root) == int(psutil.virtual_memory().total)


def test_memory_limit_fallback_to_host(tmp_path):
    # empty cgroup root -> psutil host total
    assert mem.get_memory_limit(cgroup_root=str(tmp_path)) == int(
        psutil.virtual_memory().total
    )


def test_available_memory_cgroup(tmp_path):
    root = str(tmp_path)
    _write(root, "memory.max", str(50 * GIB))
    _write(root, "memory.current", str(20 * GIB))
    assert mem.get_available_memory(cgroup_root=root) == 30 * GIB


def test_available_memory_fallback(tmp_path):
    # no usage file -> psutil available (a non-negative int)
    avail = mem.get_available_memory(cgroup_root=str(tmp_path))
    assert isinstance(avail, int) and avail >= 0


def test_get_peak_rss():
    # exact lifetime peak RSS (getrusage ru_maxrss) — positive, and monotonic non-decreasing
    p1 = mem.get_peak_rss()
    assert isinstance(p1, int) and p1 > 0
    _ = [0] * 1_000_000  # allocate a little
    assert mem.get_peak_rss() >= p1


def test_get_peak_rss_without_resource_module(monkeypatch):
    # Windows has no `resource` module (getrusage). The fallback must still return a
    # positive, monotonic non-decreasing peak rather than raising on import or call.
    monkeypatch.setattr(mem, "resource", None)
    monkeypatch.setattr(mem, "_PEAK_RSS_FALLBACK", 0)

    p1 = mem.get_peak_rss()
    assert isinstance(p1, int) and p1 > 0
    _ = [0] * 1_000_000  # allocate a little
    p2 = mem.get_peak_rss()
    assert p2 >= p1  # monotonic, like a real lifetime peak

    # a psutil failure degrades to the last known value instead of raising
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no psutil here")

    monkeypatch.setattr(mem.psutil, "Process", _Boom)
    assert mem.get_peak_rss() == p2


def test_set_process_memory_limit_applies_and_restores():
    # Linux: the limit is actually applied (read back via getrusage limits); then restored.
    import resource as _resource

    if not hasattr(_resource, "RLIMIT_DATA"):
        return  # non-Linux: covered by the no-op branch below
    old_soft, old_hard = _resource.getrlimit(_resource.RLIMIT_DATA)
    try:
        assert mem.set_process_memory_limit(4 * GIB) is True
        soft, hard = _resource.getrlimit(_resource.RLIMIT_DATA)
        assert soft == 4 * GIB
        assert hard == old_hard  # hard limit untouched — that door only closes
    finally:
        _resource.setrlimit(_resource.RLIMIT_DATA, (old_soft, old_hard))


def test_set_process_memory_limit_noop_without_resource(monkeypatch):
    monkeypatch.setattr(mem, "resource", None)
    assert mem.set_process_memory_limit(1 * GIB) is False


def test_growth_cap_is_measured_from_current_usage():
    # RLIMIT_DATA bounds TOTAL anonymous memory, so a cap of "a share of what is available"
    # would put a process already holding more than that over the line the moment it is armed.
    # The cap must say how much MORE may be used.
    import unittest.mock as um

    own = 5 * GIB
    available = 12 * GIB
    with um.patch.object(mem, "_own_data_segment", lambda: own), um.patch.object(
        mem, "get_available_memory", lambda **kw: available
    ):
        tight = mem.growth_memory_cap(divisor=6)
        wide = mem.growth_memory_cap(divisor=1)

    assert tight == own + int(0.9 * available / 6)
    assert wide == own + int(0.9 * available)
    assert tight > own  # never arms a cap the process has already exceeded
    assert wide > tight  # undivided allowance is the looser of the two


def test_growth_cap_keeps_the_container_whole():
    # N workers each growing by ratio*available/N sums to ratio*available on top of a usage
    # that is already counted, so the container cannot be pushed past its limit
    import unittest.mock as um

    limit, used, workers = 60 * GIB, 45 * GIB, 6
    available = limit - used

    with um.patch.object(mem, "_own_data_segment", lambda: 4 * GIB), um.patch.object(
        mem, "get_available_memory", lambda **kw: available
    ):
        cap = mem.growth_memory_cap(divisor=workers)

    growth_each = cap - 4 * GIB
    assert used + workers * growth_each < limit


def test_growth_cap_declines_when_there_is_no_room():
    import unittest.mock as um

    with um.patch.object(mem, "_own_data_segment", lambda: GIB), um.patch.object(
        mem, "get_available_memory", lambda **kw: 100 * 1024 * 1024
    ):
        # would fail immediately -> do not arm
        assert mem.growth_memory_cap(divisor=6) == 0
    with um.patch.object(mem, "get_available_memory", lambda **kw: 0):
        assert mem.growth_memory_cap(divisor=1) == 0


def test_memory_cap_restores_the_previous_cap_not_no_cap():
    # a tight cap nested inside a looser one must leave the looser one in force on the way out
    import resource as _resource
    import unittest.mock as um

    if not hasattr(_resource, "RLIMIT_DATA"):
        return
    soft0, hard0 = _resource.getrlimit(_resource.RLIMIT_DATA)
    try:
        with um.patch.object(mem, "get_available_memory", lambda **kw: 8 * GIB):
            with mem.memory_cap(divisor=1):  # outer, looser
                outer = _resource.getrlimit(_resource.RLIMIT_DATA)[0]
                with mem.memory_cap(divisor=4):  # inner, tighter
                    inner = _resource.getrlimit(_resource.RLIMIT_DATA)[0]
                    assert inner < outer
                assert _resource.getrlimit(_resource.RLIMIT_DATA)[0] == outer
            assert _resource.getrlimit(_resource.RLIMIT_DATA)[0] == soft0
        # the hard limit is never touched, which is what makes disarming possible
        assert _resource.getrlimit(_resource.RLIMIT_DATA)[1] == hard0
    finally:
        _resource.setrlimit(_resource.RLIMIT_DATA, (soft0, hard0))


def test_memory_cap_runs_uncapped_when_it_cannot_arm():
    import unittest.mock as um

    with um.patch.object(mem, "growth_memory_cap", lambda *a, **k: 0):
        with mem.memory_cap(divisor=6):
            pass  # must not raise


def test_memory_fail_recovery_defaults_off():
    from activitysim.core.configuration.top import Settings

    assert Settings().memory_fail_recovery is False
