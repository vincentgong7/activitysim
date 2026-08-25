# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import contextlib
import datetime
import gc
import glob
import logging
import multiprocessing
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

from activitysim.core import config, util, workflow

try:
    import resource  # Unix-only (getrusage); not available on Windows
except ImportError:
    resource = None

# high-water mark for get_peak_rss's Windows fallback (kept monotonic)
_PEAK_RSS_FALLBACK = 0

logger = logging.getLogger(__name__)

USS = True

GLOBAL_HWM = {}  # to avoid confusion with chunk local hwm

MEM_TRACE_TICK_LEN = 5
MEM_PARENT_TRACE_TICK_LEN = 15
MEM_SNOOP_TICK_LEN = 5
MEM_TICK = 0

MEM_LOG_FILE_NAME = "mem.csv"
OMNIBUS_LOG_FILE_NAME = f"omnibus_mem.csv"

SUMMARY_BIN_SIZE_IN_SECONDS = 15

mem_log_lock = threading.Lock()


def time_bin(timestamps):
    bins_size_in_seconds = SUMMARY_BIN_SIZE_IN_SECONDS
    epoch = pd.Timestamp("1970-01-01")
    seconds_since_epoch = (timestamps - epoch) // pd.Timedelta("1s")
    bin = seconds_since_epoch - (seconds_since_epoch % bins_size_in_seconds)

    return pd.to_datetime(bin, unit="s", origin="unix")


def consolidate_logs(state: workflow.State):
    """
    Consolidate and aggregate subprocess mem logs
    """

    if not state.settings.multiprocess:
        return

    delete_originals = not state.settings.keep_mem_logs
    omnibus_df = []

    # for each multiprocess step
    multiprocess_steps = state.settings.multiprocess_steps
    if multiprocess_steps is not None:
        multiprocess_steps = [i.dict() for i in multiprocess_steps]
    for step in multiprocess_steps:
        step_name = step.get("name", None)

        logger.debug(f"mem.consolidate_logs for step {step_name}")

        glob_file_name = state.get_log_file_path(
            f"{step_name}*{MEM_LOG_FILE_NAME}", prefix=False
        )
        glob_files = glob.glob(str(glob_file_name))

        if not glob_files:
            continue

        logger.debug(
            f"mem.consolidate_logs consolidating {len(glob_files)} logs for {step_name}"
        )

        # for each individual log
        step_summary_df = []
        for f in glob_files:
            df = pd.read_csv(f, comment="#")

            df = df[["rss", "uss", "event", "time"]]

            df.rss = df.rss.astype(np.int64)
            df.uss = df.uss.astype(np.int64)

            df["time"] = time_bin(
                pd.to_datetime(df.time, errors="coerce", format="%Y/%m/%d %H:%M:%S")
            )

            # consolidate events (duplicate rows should be idle steps (e.g. log_rss)
            df = (
                df.groupby("time")
                .agg(
                    rss=("rss", "max"),
                    uss=("uss", "max"),
                )
                .reset_index(drop=False)
            )

            step_summary_df.append(df)  # add step_df to step summary

        # aggregate the individual the logs into a single step log
        step_summary_df = pd.concat(step_summary_df)
        step_summary_df = (
            step_summary_df.groupby("time")
            .agg(rss=("rss", "sum"), uss=("uss", "sum"), num_files=("rss", "size"))
            .reset_index(drop=False)
        )
        step_summary_df = step_summary_df.sort_values("time")

        step_summary_df["step"] = step_name

        # scale missing values (might be missing idle steps for some chunk_tags)
        scale = (
            1
            + (len(glob_files) - step_summary_df.num_files) / step_summary_df.num_files
        )
        for c in ["rss", "uss"]:
            step_summary_df[c] = (step_summary_df[c] * scale).astype(np.int64)

        step_summary_df["scale"] = scale
        del step_summary_df["num_files"]  # do we want to keep track of scale factor?

        if delete_originals:
            util.delete_files(glob_files, f"mem.consolidate_logs.{step_name}")

        # write aggregate step log
        output_path = state.get_log_file_path(f"mem_{step_name}.csv", prefix=False)
        logger.debug(
            f"chunk.consolidate_logs writing step summary log for step {step_name} to {output_path}"
        )
        step_summary_df.to_csv(output_path, mode="w", index=False)

        omnibus_df.append(step_summary_df)  # add step summary to omnibus

    # aggregate the step logs into a single omnibus log ordered by timestamp
    omnibus_df = pd.concat(omnibus_df)
    omnibus_df = omnibus_df.sort_values("time")

    output_path = state.get_log_file_path(OMNIBUS_LOG_FILE_NAME, prefix=False)
    logger.debug(f"chunk.consolidate_logs writing omnibus log to {output_path}")
    omnibus_df.to_csv(output_path, mode="w", index=False)


def check_global_hwm(tag, value, label):
    assert value is not None

    hwm = GLOBAL_HWM.setdefault(tag, {})

    is_new_hwm = value > hwm.get("mark", 0) or not hwm
    if is_new_hwm:
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        hwm["mark"] = value
        hwm["timestamp"] = timestamp
        hwm["label"] = label

    return is_new_hwm


def log_global_hwm():
    process_name = multiprocessing.current_process().name

    for tag in GLOBAL_HWM:
        hwm = GLOBAL_HWM[tag]
        value = hwm.get("mark", 0)
        logger.info(
            f"{process_name} high water mark {tag}: {util.INT(value)} ({util.GB(value)}) "
            f"timestamp: {hwm.get('timestamp', '<none>')} label:{hwm.get('label', '<none>')}"
        )


def trace_memory_info(event, trace_ticks=0, force_garbage_collect=False, *, state):
    global MEM_TICK

    if state is None:
        raise ValueError("state cannot be None")

    tick = time.time()
    if trace_ticks and (tick - MEM_TICK < trace_ticks):
        return
    MEM_TICK = tick

    if force_garbage_collect:
        was_disabled = not gc.isenabled()
        if was_disabled:
            gc.enable()
        gc.collect()
        if was_disabled:
            gc.disable()

    process_name = multiprocessing.current_process().name
    pid = os.getpid()

    current_process = psutil.Process()

    if USS:
        try:
            info = current_process.memory_full_info()
            uss = info.uss
        except (PermissionError, psutil.AccessDenied, RuntimeError):
            info = current_process.memory_info()
            uss = 0
    else:
        info = current_process.memory_info()
        uss = 0

    full_rss = rss = info.rss

    num_children = 0
    for child in current_process.children(recursive=True):
        try:
            child_info = child.memory_info()
            full_rss += child_info.rss
            num_children += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            pass

    noteworthy = (
        True  # any reason not to always log this if we are filtering idle ticks?
    )

    noteworthy = (num_children > 0) or noteworthy
    noteworthy = check_global_hwm("rss", full_rss or rss, event) or noteworthy
    noteworthy = check_global_hwm("uss", uss, event) or noteworthy

    if noteworthy:
        # logger.debug(f"trace_memory_info {event} "
        #              f"rss: {GB(full_rss) if num_children else GB(rss)} "
        #              f"uss: {GB(rss)} ")

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")  # sortable

        with mem_log_lock:
            MEM_LOG_HEADER = "process,pid,rss,full_rss,uss,event,children,time"
            log_file = state.filesystem.open_log_file(
                MEM_LOG_FILE_NAME,
                "a",
                header=MEM_LOG_HEADER,
                prefix=state.get("log_file_prefix", None),
            )

            with log_file:
                print(
                    f"{process_name},"
                    f"{pid},"
                    f"{util.INT(rss)},"  # want these as ints so we can plot them...
                    f"{util.INT(full_rss)},"
                    f"{util.INT(uss)},"
                    f"{event},"
                    f"{num_children},"
                    f"{timestamp}",
                    file=log_file,
                )

    # return rss and uss for optional use by interested callers
    return full_rss or rss, uss


def get_rss(force_garbage_collect=False, uss=False):
    if force_garbage_collect:
        was_disabled = not gc.isenabled()
        if was_disabled:
            gc.enable()
        gc.collect()
        if was_disabled:
            gc.disable()

    if uss:
        try:
            info = psutil.Process().memory_full_info()
            return info.rss, info.uss
        except (PermissionError, psutil.AccessDenied, RuntimeError):
            info = psutil.Process().memory_info()
            return info.rss, 0
    else:
        info = psutil.Process().memory_info()
        return info.rss, 0


# --- real memory-ceiling introspection (cgroup-aware) ----------------------------------------------
# psutil reports the HOST's RAM, which is wrong inside a container: the process is bounded by its
# cgroup memory limit (e.g. a Kubernetes pod limit), not the node's total RAM. Chunk sizing that
# targets host RAM will overshoot the cgroup limit and get OOM-killed. These helpers read the real
# ceiling from the cgroup (v2, then v1), falling back to psutil when not containerized/unlimited.

# cgroup "unlimited" is reported as a huge sentinel; treat anything at/above it as no-limit.
_CGROUP_UNLIMITED = 0x7FFFFFFFFFFFF000  # ~9.2e18


def _read_cgroup_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _finite_limit(raw):
    """Parse a cgroup limit string; return an int only if it is a real finite limit."""
    if raw is None or raw == "max":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if 0 < n < _CGROUP_UNLIMITED else None


def get_memory_limit(cgroup_root: str = "/sys/fs/cgroup") -> int | None:
    """This process's hard memory ceiling in bytes, or None if it can't be determined.

    Prefers the cgroup limit (what actually OOM-kills us in a container) over host RAM. Tries cgroup
    v2 (``memory.max``), then cgroup v1 (``memory/memory.limit_in_bytes``), then psutil total RAM.
    """
    limit = _finite_limit(_read_cgroup_file(os.path.join(cgroup_root, "memory.max")))
    if limit is not None:
        return limit
    for rel in ("memory/memory.limit_in_bytes", "memory.limit_in_bytes"):
        limit = _finite_limit(_read_cgroup_file(os.path.join(cgroup_root, rel)))
        if limit is not None:
            return limit
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def get_available_memory(cgroup_root: str = "/sys/fs/cgroup") -> int | None:
    """Best-effort bytes still available before this process hits its ceiling.

    Uses (cgroup limit - cgroup current usage) when containerized, else psutil available RAM. Note
    cgroup ``memory.current`` counts reclaimable page cache as used, so this under-estimates the truly
    available memory — which is the safe direction for chunk sizing (errs toward smaller chunks).
    """
    limit = get_memory_limit(cgroup_root)
    used = None
    raw = _read_cgroup_file(os.path.join(cgroup_root, "memory.current"))
    if raw is not None:
        try:
            used = int(raw)
        except ValueError:
            used = None
    if used is None:
        for rel in ("memory/memory.usage_in_bytes", "memory.usage_in_bytes"):
            raw = _read_cgroup_file(os.path.join(cgroup_root, rel))
            if raw is not None:
                try:
                    used = int(raw)
                    break
                except ValueError:
                    used = None
    if limit is not None and used is not None:
        return max(0, limit - used)
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return limit


def get_peak_rss() -> int:
    """Exact lifetime peak RSS of this process in bytes, from the kernel (``getrusage`` ru_maxrss).

    Unlike the MemMonitor's periodically-sampled high-water mark, this never misses a short-lived
    transient allocation spike (a common cause of adaptive chunking under-estimating a chunk's true
    peak). Linux reports ru_maxrss in kilobytes; macOS/BSD report bytes. On Windows (no ``resource``
    module) there is no getrusage peak, so this tracks a sampled high-water mark of the current RSS,
    which keeps it monotonic non-decreasing like the real peak."""
    if resource is None:
        # Windows: no getrusage; track a sampled high-water mark of current RSS so the result stays
        # monotonic non-decreasing.
        global _PEAK_RSS_FALLBACK
        try:
            rss = int(psutil.Process().memory_info().rss)
        except Exception:
            return _PEAK_RSS_FALLBACK
        _PEAK_RSS_FALLBACK = max(_PEAK_RSS_FALLBACK, rss)
        return _PEAK_RSS_FALLBACK
    try:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ValueError, OSError):
        return 0
    return int(maxrss) * 1024 if sys.platform.startswith("linux") else int(maxrss)


# Fraction of the memory still AVAILABLE that a scope is allowed to grow into when
# memory_fail_recovery is enabled. The remainder is left as slack for everything the process
# does not account for.
#
# A cap is armed for the scope where it can do something useful and lifted again on the way
# out, rather than set once for the life of the worker. Two scopes matter:
#
#   * inside a chunk loop the allowance is divided by the worker count, because every worker
#     is doing chunked work at the same time and they share one ceiling. Keeping each
#     worker's growth inside its share is what makes a failure small enough for the chunk
#     retry to absorb.
#   * outside one, the allowance is not divided. Work there cannot be split and retried, so a
#     cap that binds it can only turn a working run into a failing one -- which is exactly
#     what a lifetime cap did on a Rotterdam-scale run: a 3.75M-row join failed for want of
#     1.29 GB while 15 GB of the container sat unused. An undivided allowance still catches
#     the case worth catching, a single outsized allocation, and turns what is otherwise a
#     silent container kill into an exception naming the step that caused it.
WORKER_MEMORY_CAP_RATIO = 0.9

# Arming a cap that leaves almost no room makes the next allocation fail whatever its size,
# and no amount of halving recovers from that. Below this much growth we decline to arm and
# say so, rather than guarantee a failure.
MIN_CAP_ALLOWANCE = 256 * 1024 * 1024


def _own_data_segment() -> int | None:
    """This process's anonymous footprint in bytes — the quantity RLIMIT_DATA meters."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def growth_memory_cap(divisor: int = 1) -> int:
    """A total-memory cap that lets this process grow by a share of what is still available.

    ``RLIMIT_DATA`` bounds a process's TOTAL anonymous memory, not its growth, so a cap of
    "a share of what is available" would put a process that already holds more than that over
    the line the moment it is armed: every later allocation fails, and halving cannot help
    because what is already held is data the work needs. The cap is therefore measured from
    what the process holds right now, so it expresses how much MORE it may use.

    That costs the container nothing, because memory already held is already counted in the
    container's current usage: N workers each growing by ratio*available/N sums to
    ratio*available on top of a usage that is already there.

    Returns 0 when no cap should be armed — no reading available, or too little room left for
    a cap to be anything but a guaranteed failure.
    """
    own = _own_data_segment()
    available = get_available_memory()
    if own is None or not available:
        return 0
    allowance = int(WORKER_MEMORY_CAP_RATIO * available / max(int(divisor), 1))
    if allowance < MIN_CAP_ALLOWANCE:
        logger.warning(
            f"memory_fail_recovery: not arming a cap — only {util.GB(allowance)} of growth "
            f"would be allowed, which would fail immediately. Consider fewer workers or a "
            f"larger memory limit."
        )
        return 0
    return own + allowance


def step_memory_cap(state, step_name: str = None):
    """The loose, per-model-step ceiling — a diagnostic rather than a guarantee.

    Undivided by worker count on purpose. Everything a step does outside a chunk loop --
    building the chooser table, joining it to the sampled alternatives, concatenating the
    chunked results -- has no smaller form to retry, so a cap sized to one worker's share
    can only convert work that would have completed into a failure. Measured against the
    whole of what is available, normal work never notices it.

    What it does catch is one step trying to take far more than the machine has. Today that
    ends as a container kill: every log stops in the same second and nothing records which
    worker or which model was responsible. Under this ceiling the same event raises where it
    happened, with a traceback naming the step and the size of the allocation. The run still
    fails; it fails with an explanation.

    It cannot prevent every kill. Several workers each taking a little too much still add up
    past the limit, because each of them measures the same available memory. That case needs
    the divided cap, which is armed around chunked work where a failure can be absorbed.
    """
    settings = getattr(state, "settings", None) if state is not None else None
    if not getattr(settings, "memory_fail_recovery", False):
        return contextlib.nullcontext()
    return memory_cap(divisor=1, trace_label=step_name)


@contextlib.contextmanager
def memory_cap(divisor: int = 1, trace_label: str = None):
    """Cap this process's anonymous memory for the duration of the block, then restore it.

    Restores the PREVIOUS soft limit rather than removing the cap, so a tighter cap nested
    inside a looser one leaves the looser one in force on the way out. Recomputed on every
    entry: usage and availability move over a run, and a value computed once at startup is
    stale by the time it is needed.

    Never raises on account of the cap itself — if the limit cannot be read or applied the
    block simply runs uncapped.
    """
    nbytes = growth_memory_cap(divisor)
    previous = None
    hard = None
    if nbytes and resource is not None and hasattr(resource, "RLIMIT_DATA"):
        try:
            previous, hard = resource.getrlimit(resource.RLIMIT_DATA)
            resource.setrlimit(resource.RLIMIT_DATA, (nbytes, hard))
            logger.debug(
                f"{trace_label or 'scope'}: memory cap armed at {util.GB(nbytes)} "
                f"(1/{divisor} of available)"
            )
        except (ValueError, OSError) as e:
            logger.warning(f"could not arm memory cap: {e}")
            previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (previous, hard))
            except (ValueError, OSError) as e:
                logger.warning(f"could not restore the previous memory cap: {e}")


def set_process_memory_limit(nbytes: int) -> bool:
    """Cap this process's anonymous memory at ``nbytes`` (Linux only).

    Sets ``RLIMIT_DATA``, which covers the heap and anonymous mappings but NOT file-backed
    mappings (memory-mapped skims stay usable regardless of the cap). An allocation that
    would exceed the cap then raises a catchable ``MemoryError`` in this process instead of
    growing the cgroup toward its limit, where the kernel would kill every process in the
    container as a group (``memory.oom.group``).

    Returns True if the limit was applied. On platforms without ``resource``/``RLIMIT_DATA``
    (Windows) this is a no-op returning False: there the allocator already refuses
    over-commitment at allocation time with a MemoryError, which is the behavior this
    function exists to create.
    """
    if resource is None or not hasattr(resource, "RLIMIT_DATA"):
        logger.info("set_process_memory_limit: not supported on this platform (no-op)")
        return False
    try:
        # set only the SOFT limit and leave the hard limit untouched: the soft limit is
        # what makes allocations fail, and an unprivileged process can adjust it freely,
        # whereas lowering the hard limit is a one-way door (raising it back requires
        # CAP_SYS_RESOURCE).
        _, hard = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (int(nbytes), hard))
    except (ValueError, OSError) as e:
        logger.warning(f"set_process_memory_limit: could not set RLIMIT_DATA: {e}")
        return False
    logger.info(f"set_process_memory_limit: RLIMIT_DATA = {util.GB(int(nbytes))}")
    return True


def shared_memory_size(data_buffers):
    """
    return total size of the multiprocessing shared memory block in data_buffers

    Returns
    -------

    """

    shared_size = 0

    if data_buffers is None:
        data_buffers = {}

    for k, data_buffer in data_buffers.items():
        if isinstance(data_buffer, str) and data_buffer.startswith("sh.Dataset:"):
            from sharrow import Dataset

            shared_size += Dataset.shm.preload_shared_memory_size(data_buffer[11:])
            continue
        try:
            obj = data_buffer.get_obj()
        except Exception:
            obj = data_buffer
        data = np.ctypeslib.as_array(obj)
        data_size = data.nbytes

        shared_size += data_size

    return shared_size
