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


def test_available_memory_can_ignore_reclaimable_page_cache(tmp_path):
    """A container that looks full of page cache is not full.

    A run mmaps its skims and writes tens of GB of output, so memory.current ends up dominated
    by file-backed cache the kernel evicts on demand. Measured that way a 60 GB container looks
    like it has 2 GB left while only 6 GB is genuinely unreclaimable. A ceiling sized on the
    first number refuses work that would have fitted -- which is how a full-population run lost
    write_data_dictionary to a 2 GB allocation while holding 0.77 GB.
    """
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(root, "memory.current", str(58 * GIB))          # 52 GiB of it is page cache
    _write(root, "memory.stat", "anon 5368709120\nfile 55834574848\nshmem 1073741824\n")

    assert mem.get_nonreclaimable_used(cgroup_root=root) == 6 * GIB
    assert mem.get_available_memory(cgroup_root=root) == 2 * GIB
    assert (
        mem.get_available_memory(cgroup_root=root, basis=mem.BASIS_NONRECLAIMABLE)
        == 54 * GIB
    )


def test_nonreclaimable_falls_back_to_memory_current_when_unreadable(tmp_path):
    # no memory.stat -> the caller must still get the old basis rather than nothing
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(root, "memory.current", str(58 * GIB))
    assert mem.get_nonreclaimable_used(cgroup_root=root) is None
    assert (
        mem.get_available_memory(cgroup_root=root, basis=mem.BASIS_NONRECLAIMABLE)
        == 2 * GIB
    )


def test_step_cap_measures_against_nonreclaimable_memory(monkeypatch):
    """The step ceiling must not close in as page cache accumulates."""
    seen = {}

    def fake_available(cgroup_root="/sys/fs/cgroup", basis=mem.BASIS_USAGE):
        seen["basis"] = basis
        return 40 * GIB

    monkeypatch.setattr(mem, "get_available_memory", fake_available)
    monkeypatch.setattr(mem, "_own_data_segment", lambda: 2 * GIB)

    class _S:
        memory_fail_recovery = True
        memory_step_cap_ratio = 1.0

    class _State:
        settings = _S()

    with mem.step_memory_cap(_State(), "write_data_dictionary"):
        pass
    assert seen["basis"] == mem.BASIS_NONRECLAIMABLE

    # a bare growth cap still defaults to the legacy basis; callers opt in
    seen.clear()
    mem.growth_memory_cap(divisor=6)
    assert seen["basis"] == mem.BASIS_USAGE


def test_working_set_keeps_active_cache_and_releases_written_pages(tmp_path):
    """The two kinds of page cache a run accumulates must not be treated alike.

    The mapped skims are read over and over, so their pages stay ACTIVE and are a real cost that
    the next chunk will need again. The output tables are written once and never read; those pages
    go INACTIVE and the kernel drops them on demand. memory.current charges for both, so the chunk
    budget shrank as the run wrote its results — pressure that chunking cannot relieve and should
    not respond to.
    """
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(root, "memory.current", str(58 * GIB))
    _write(
        root,
        "memory.stat",
        "anon 5368709120\n"           # 5 GiB private
        "shmem 1073741824\n"          # 1 GiB shm
        "file 55834574848\n"          # 52 GiB of cache, split below
        "active_file 4294967296\n"    # 4 GiB skim working set — must stay counted
        "inactive_file 51539607552\n",  # 48 GiB written output — must be released
    )
    assert mem.get_working_set_used(cgroup_root=root) == 10 * GIB
    assert mem.get_available_memory(cgroup_root=root) == 2 * GIB
    assert (
        mem.get_available_memory(cgroup_root=root, basis=mem.BASIS_WORKING_SET) == 50 * GIB
    )
    # strictly between the two extremes: it gives back less than nonreclaimable does
    assert mem.get_available_memory(
        cgroup_root=root, basis=mem.BASIS_WORKING_SET
    ) < mem.get_available_memory(cgroup_root=root, basis=mem.BASIS_NONRECLAIMABLE)


def test_working_set_falls_back_when_the_cache_split_is_missing(tmp_path):
    # cgroup v1 without the active/inactive split -> caller must still get the legacy basis
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(root, "memory.current", str(58 * GIB))
    _write(root, "memory.stat", "anon 5368709120\nshmem 1073741824\n")
    assert mem.get_working_set_used(cgroup_root=root) is None
    assert (
        mem.get_available_memory(cgroup_root=root, basis=mem.BASIS_WORKING_SET) == 2 * GIB
    )


class _FakeVM:
    def __init__(self, total, available):
        self.total, self.available = total, available


def _fake_psutil(monkeypatch, total, available, physical=None, logical=None):
    """Stand in for psutil so a test can pose as a machine it is not running on."""

    class _P:
        @staticmethod
        def virtual_memory():
            return _FakeVM(total, available)

        @staticmethod
        def cpu_count(logical=True):  # noqa: A002 — psutil's own parameter name
            return _P._logical if logical else _P._physical

    _P._physical, _P._logical = physical, logical
    monkeypatch.setattr(mem, "psutil", _P)
    return _P


def test_recommend_num_processes_in_a_container(tmp_path, monkeypatch):
    """The container case: memory from the cgroup working set, cores from the cgroup quota."""
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(root, "memory.current", str(50 * GIB))          # mostly written-out cache
    _write(
        root,
        "memory.stat",
        "anon 6442450944\nshmem 1073741824\n"             # 6 + 1 GiB
        "active_file 1073741824\ninactive_file 45097156608\n",
    )
    # 60 - 8 = 52 GiB available; x 0.5 = 26 GiB; / 3 GiB = 8 workers, under the 12-CPU quota
    _write(root, "cpu.max", "1200000 100000")
    assert mem.get_cpu_limit(cgroup_root=root) == 12
    n = mem.recommend_num_processes(
        target_per_worker=3 * GIB, safety=0.5, cgroup_root=root
    )
    assert n == 8


def test_recommend_num_processes_is_clamped_by_the_cpu_quota(tmp_path):
    """A pod with plenty of memory but few CPUs must not be told to run a worker per GB.

    psutil would report the host's processors here, which is exactly the number a container must
    not use.
    """
    root = str(tmp_path)
    _write(root, "memory.max", str(60 * GIB))
    _write(
        root,
        "memory.stat",
        "anon 2147483648\nshmem 0\nactive_file 0\ninactive_file 0\n",
    )
    _write(root, "cpu.max", "400000 100000")               # 4 CPUs
    # memory alone would allow (60-2) * 0.5 / 3 = 9 workers
    assert mem.recommend_num_processes(3 * GIB, 0.5, cgroup_root=root) == 4


def test_recommend_num_processes_outside_a_container(tmp_path, monkeypatch):
    """Bare metal, and Windows: no cgroup, so the OS's own available-memory figure decides.

    On Linux that is MemAvailable and on Windows AvailPhys; both already report memory obtainable
    without going to disk, so no per-platform branch is needed here. Physical cores bound it,
    because workers are compute-bound and SMT siblings are not extra places to put one.
    """
    _fake_psutil(
        monkeypatch, total=64 * GIB, available=32 * GIB, physical=8, logical=16
    )
    # no cgroup files under tmp_path -> the psutil path; 32 x 0.5 / 3 = 5 workers, under 8 cores
    assert mem.recommend_num_processes(3 * GIB, 0.5, cgroup_root=str(tmp_path)) == 5


def test_recommend_num_processes_on_a_small_machine(tmp_path, monkeypatch):
    """A laptop with little free memory gets one worker, never zero."""
    _fake_psutil(
        monkeypatch, total=8 * GIB, available=2 * GIB, physical=4, logical=8
    )
    assert mem.recommend_num_processes(3 * GIB, 0.5, cgroup_root=str(tmp_path)) == 1


def test_recommend_num_processes_without_a_physical_core_count(tmp_path, monkeypatch):
    # psutil returns None for the physical count on some platforms -> fall back to logical
    _fake_psutil(
        monkeypatch, total=64 * GIB, available=64 * GIB, physical=None, logical=3
    )
    assert mem.recommend_num_processes(3 * GIB, 1.0, cgroup_root=str(tmp_path)) == 3


def test_recommend_num_processes_declines_rather_than_guessing(tmp_path, monkeypatch):
    """Unreadable machine -> None, so the caller keeps ActivitySim's own default."""

    class _Broken:
        @staticmethod
        def virtual_memory():
            raise RuntimeError("no")

        @staticmethod
        def cpu_count(logical=True):
            return None

    monkeypatch.setattr(mem, "psutil", _Broken)
    monkeypatch.setattr(mem.os, "cpu_count", lambda: None)
    assert mem.recommend_num_processes(3 * GIB, 0.5, cgroup_root=str(tmp_path)) is None
    # and a nonsensical target is declined too, rather than dividing by zero
    _fake_psutil(monkeypatch, total=64 * GIB, available=32 * GIB, physical=8, logical=8)
    assert mem.recommend_num_processes(0, 0.5, cgroup_root=str(tmp_path)) is None


def test_worker_memory_target_defaults_to_the_builtin(monkeypatch):
    from activitysim.core.configuration.top import Settings

    assert Settings().worker_memory_target == 0        # 0 means "use the built-in"
    assert mem.DEFAULT_WORKER_MEMORY_TARGET == 3 * GIB


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


def test_step_cap_defaults_to_the_whole_of_what_is_available():
    # a diagnostic ceiling must be a fixed line, not one that closes in as memory runs low
    import unittest.mock as um

    from activitysim.core.configuration.top import Settings

    assert Settings().memory_step_cap_ratio == 1.0

    own, available = 10 * GIB, 19 * GIB
    with um.patch.object(mem, "_own_data_segment", lambda: own), um.patch.object(
        mem, "get_available_memory", lambda **kw: available
    ):
        at_default = mem.growth_memory_cap(divisor=1, ratio=1.0)
        tightened = mem.growth_memory_cap(divisor=1, ratio=0.9)

    # at 1.0 the step may grow into everything that is left -- i.e. up to the container limit
    assert at_default == own + available
    # anything less holds it short of memory the container would have given it
    assert tightened < at_default


def test_memory_fail_recovery_defaults_off():
    from activitysim.core.configuration.top import Settings

    assert Settings().memory_fail_recovery is False
