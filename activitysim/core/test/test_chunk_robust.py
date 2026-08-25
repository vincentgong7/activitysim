# ActivitySim
# See full license in LICENSE.txt.
"""Tests for the cgroup-aware ``chunk_size_mode: auto`` budget and the adaptive sizing it feeds.
See also test_mem.py for the cgroup / worker-count helpers."""
from __future__ import annotations

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import chunk, mem, simulate, workflow

TESTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(TESTDIR, "data")

GIB = 1024**3


@pytest.fixture
def state() -> workflow.State:
    st = workflow.State()
    st.initialize_filesystem(
        working_dir=TESTDIR, data_dir=(DATADIR,)
    ).default_settings()
    st.settings.check_for_variability = False
    return st


@pytest.fixture
def spec(state):
    return state.filesystem.read_model_spec(file_name="sample_spec.csv")


@pytest.fixture
def data():
    return pd.read_csv(os.path.join(DATADIR, "data.csv"))


EXPECTED = pd.Series([1, 1, 1])


def test_resolve_chunk_size_fixed_is_legacy(state):
    # default (fixed) mode must return the static chunk_size verbatim -> no behavior change
    state.settings.chunk_size = 123456
    assert chunk.resolve_chunk_size(state) == 123456


def test_resolve_chunk_size_auto(state):
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 0.75
    limit = mem.get_memory_limit()
    budget = chunk.resolve_chunk_size(state)
    # Budget = safety_factor * AVAILABLE memory (limit - current usage): strictly positive, and never
    # above safety_factor * the real ceiling (available <= limit).
    assert budget >= 1
    assert budget <= max(1, int(limit * 0.75))


def test_resolve_chunk_size_auto_safety_factor_scales(state):
    # a smaller safety_factor yields a smaller (or equal) budget
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 0.75
    hi = chunk.resolve_chunk_size(state)
    state.settings.chunk_size_safety_factor = 0.25
    lo = chunk.resolve_chunk_size(state)
    assert lo <= hi


def test_resolve_chunk_size_auto_zero_headroom_is_not_full_limit(state, monkeypatch):
    # available == 0 legitimately means "no headroom" and must NOT be treated as "unknown" and replaced
    # with the full limit (that would hand out a huge budget exactly when memory is exhausted).
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 100 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 0)
    budget = chunk.resolve_chunk_size(state)
    assert budget < GIB  # tiny (floored to keep chunking on), NOT ~75 GB


def test_resolve_chunk_size_auto_divides_by_workers(state, monkeypatch):
    # multiprocess: the shared budget is divided by the per-step worker count (num_processes injectable)
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 1.0
    state.settings.multiprocess = True
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 40 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 40 * GIB)
    state.add_injectable("num_processes", 4)
    budget = chunk.resolve_chunk_size(state)
    assert budget == 10 * GIB  # 40 GB / 4 workers


def test_auto_mode_simple_simulate_matches_fixed(state, data, spec):
    # auto mode must produce the same choices as the legacy path
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_growth_cap = 2.0
    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    pdt.assert_series_equal(choices.reset_index(drop=True), EXPECTED, check_dtype=False)


def test_auto_mode_splits_into_multiple_chunks(state, data, monkeypatch):
    # Force a tiny auto budget so the choosers are split into MULTIPLE chunks (not a single-chunk run),
    # exercising the auto chunk-sizing loop. Assert the chunks partition the choosers exactly.
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    state.settings.default_initial_rows_per_chunk = 1  # tiny first (probe) chunk
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 1000)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 1)

    chunks = [
        chooser_chunk.copy()
        for _i, chooser_chunk, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_multichunk"
        )
    ]
    assert len(chunks) > 1  # the tiny budget forced more than one chunk
    # chunks partition the original choosers exactly (rows + order preserved, none lost/duplicated)
    pdt.assert_frame_equal(pd.concat(chunks), data)


def test_chunk_memory_settings_validation():
    # the auto-mode knobs reject nonsensical values at configuration time
    from pydantic import ValidationError

    from activitysim.core.configuration.top import Settings

    Settings(chunk_size_safety_factor=0.5)  # in (0, 1] — ok
    Settings(chunk_growth_cap=1.5)  # off (0) or >= 1 — ok
    Settings(chunk_row_size_margin=1.3)  # >= 1 — ok
    with pytest.raises(ValidationError):
        Settings(chunk_size_safety_factor=0.0)
    with pytest.raises(ValidationError):
        Settings(chunk_size_safety_factor=1.5)
    with pytest.raises(ValidationError):
        Settings(chunk_growth_cap=0.5)  # would shrink every chunk toward collapse
    with pytest.raises(ValidationError):
        Settings(chunk_row_size_margin=0.9)
    Settings(chunk_peak_backoff_ratio=0.8)  # in (0, 1] — ok
    with pytest.raises(ValidationError):
        Settings(chunk_peak_backoff_ratio=0.0)
    with pytest.raises(ValidationError):
        Settings(chunk_peak_backoff_ratio=1.5)


def test_auto_growth_cap_default():
    # auto mode caps growth by default; fixed keeps legacy uncapped; explicit setting wins
    from activitysim.core.configuration.top import Settings

    assert chunk._effective_growth_cap(Settings(chunk_size_mode="fixed")) == 0
    assert (
        chunk._effective_growth_cap(Settings(chunk_size_mode="auto"))
        == chunk.AUTO_DEFAULT_GROWTH_CAP
    )
    assert (
        chunk._effective_growth_cap(
            Settings(chunk_size_mode="auto", chunk_growth_cap=3.0)
        )
        == 3.0
    )


def test_auto_probe_chunk_is_capped(state, monkeypatch):
    # with no cached row_size, the first (probe) chunk under auto is capped at
    # MAX_AUTO_PROBE_ROWS even when default_initial_rows_per_chunk is huge
    n = chunk.MAX_AUTO_PROBE_ROWS * 3
    data = pd.DataFrame({"x": range(n)})
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    state.settings.default_initial_rows_per_chunk = (
        50_000  # deliberately oversized probe
    )
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 100 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 100 * GIB)

    sizes = [
        len(chooser_chunk)
        for _i, chooser_chunk, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_probe_cap"
        )
    ]
    assert sizes[0] <= chunk.MAX_AUTO_PROBE_ROWS


def test_chunking_settings_logged_once(state, caplog):
    # the audit line contains every effective chunking parameter, once per process
    import logging

    chunk._CHUNK_SETTINGS_LOGGED = False
    state.settings.chunk_size_mode = "auto"
    with caplog.at_level(logging.INFO, logger="activitysim.core.chunk"):
        chunk.log_chunking_settings(state)
        chunk.log_chunking_settings(state)  # second call must be a no-op
    msgs = [r.message for r in caplog.records if "chunking settings:" in r.message]
    assert len(msgs) == 1
    for key in (
        "chunk_size_mode=auto",
        "chunk_size=",
        "chunk_size_safety_factor=",
        "chunk_growth_cap=",
        "(effective=",
        "chunk_row_size_margin=",
        "chunk_training_mode=",
        "chunk_method=",
        "default_initial_rows_per_chunk=",
        "auto probe cap=",
        "num_processes=",
        "chunk_peak_backoff_ratio=",
    ):
        assert key in msgs[0], key
    chunk._CHUNK_SETTINGS_LOGGED = False  # don't leak state to other tests


def test_auto_overrides_passed_static_chunk_size(state, monkeypatch):
    # callers historically pass settings.chunk_size down explicitly; under auto the
    # runtime-resolved budget must win or parts of the pipeline silently run static
    data = pd.DataFrame({"x": range(100)})
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 10 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 10 * GIB)
    budgets = [
        getattr(sizer, "base_chunk_size", None) or sizer.chunk_size
        for _i, _c, _label, sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_override", chunk_size=999 * GIB
        )
    ]
    assert budgets and all(
        b <= 5 * GIB for b in budgets
    )  # resolved from the 10 GiB limit, not the passed 999 GiB

    # chunk_size == 0 is the deliberate "run this component chunkless" signal — auto must
    # NOT override it (tour scheduling logsums relies on this)
    chunks = [
        c
        for _i, c, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_keeps_chunkless", chunk_size=0
        )
    ]
    assert len(chunks) == 1 and len(chunks[0]) == len(data)


def test_run_with_memory_retry_splits_and_merges():
    # first call on the full chunk fails; halves succeed; results come back in row order
    data = pd.DataFrame({"x": range(100)})
    calls = []

    def work(chunk_df):
        calls.append(len(chunk_df))
        if len(chunk_df) > 60:
            raise MemoryError("too big")
        return chunk_df["x"].sum()

    out = chunk.run_with_memory_retry(work, data, trace_label="t")
    assert calls == [100, 50, 50]  # full attempt, then the two halves
    assert sum(out) == data["x"].sum()  # nothing lost, nothing duplicated


def test_run_with_memory_retry_exhaustion_reraises():
    data = pd.DataFrame({"x": range(64)})

    def always_fails(chunk_df):
        raise MemoryError("always")

    with pytest.raises(MemoryError):
        chunk.run_with_memory_retry(always_fails, data, trace_label="t")


def test_run_with_memory_retry_success_passthrough():
    # no failure -> exactly one call, one result
    data = pd.DataFrame({"x": range(10)})
    out = chunk.run_with_memory_retry(lambda df: len(df), data)
    assert out == [10]


def _choosers_and_sampled_alts(n=8):
    """A chooser table and a sampled-alternatives table shaped like the real ones:
    one row per (chooser, alternative), indexed by chooser id, varying counts per
    chooser, laid out in chooser order."""
    choosers = pd.DataFrame(
        {"c": range(n)}, index=pd.Index(range(n), name="chooser_id")
    )
    alt_index = []
    for chooser_id in choosers.index:
        alt_index += [chooser_id] * (chooser_id % 3 + 1)
    alts = pd.DataFrame(
        {"a": range(len(alt_index))}, index=pd.Index(alt_index, name="chooser_id")
    )
    return choosers, alts


def test_run_with_memory_retry_alts_keeps_alts_with_their_choosers():
    # the alternatives must follow their choosers into the halves; cutting them in half
    # independently would mis-pair them
    choosers, alts = _choosers_and_sampled_alts(8)
    chooser_counts = []

    def work(chunk_df, alt_df):
        chooser_counts.append(len(chunk_df))
        if len(chunk_df) > 4:
            raise MemoryError("too big")
        # no alternative may arrive without its chooser, and none may go missing
        assert set(alt_df.index) == set(chunk_df.index)
        return alt_df["a"].tolist()

    out = chunk.run_with_memory_retry_alts(work, choosers, alts, trace_label="t")

    assert chooser_counts == [8, 4, 4]  # full attempt, then the two halves
    # every alternative row appears exactly once, in the original order
    assert [a for part in out for a in part] == alts["a"].tolist()


def test_run_with_memory_retry_alts_success_passthrough():
    choosers, alts = _choosers_and_sampled_alts(6)
    out = chunk.run_with_memory_retry_alts(
        lambda df, alt_df: (len(df), len(alt_df)), choosers, alts
    )
    assert out == [(len(choosers), len(alts))]


def _rng_and_state(persons):
    """A live random channel for `persons`, plus a minimal stand-in for the state object
    the retry helper uses to reach it."""
    from types import SimpleNamespace

    from activitysim.core import random as asim_random

    rng = asim_random.Random()
    rng.set_base_seed(0)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)
    return rng, SimpleNamespace(get_rn_generator=lambda: rng)


def _persons(n=8):
    return pd.DataFrame(
        {"x": range(n)}, index=pd.Index(range(1, n + 1), name="person_id")
    )


def test_splitting_a_chunk_does_not_change_random_draws():
    # each row is seeded from its own index, so halving a chunk must be invisible to the
    # random streams. Everything else about the retry depends on this holding.
    persons = _persons()
    whole, _ = _rng_and_state(persons)
    reference = whole.random_for_df(persons)

    halves_rng, _ = _rng_and_state(persons)
    halves = np.concatenate(
        [
            halves_rng.random_for_df(persons.iloc[:4]),
            halves_rng.random_for_df(persons.iloc[4:]),
        ]
    )
    npt.assert_array_equal(reference, halves)


def test_retry_reproduces_draws_when_the_failed_attempt_already_drew():
    # A failed attempt that got as far as drawing has advanced each row's offset. Without
    # rewinding, the retry would continue the random stream instead of repeating it, and the
    # run's results would depend on whether a MemoryError happened to occur.
    persons = _persons()

    def make_work(rng, fail_first):
        calls = {"n": 0}

        def work(chunk_df):
            calls["n"] += 1
            drawn = rng.random_for_df(chunk_df)
            if calls["n"] == 1 and fail_first:
                raise MemoryError("failed after drawing")
            return drawn

        return work

    rng, state = _rng_and_state(persons)
    reference = np.concatenate(
        chunk.run_with_memory_retry(make_work(rng, False), persons, state=state)
    )

    rng, state = _rng_and_state(persons)
    retried = np.concatenate(
        chunk.run_with_memory_retry(
            make_work(rng, True), persons, state=state, trace_label="t"
        )
    )

    npt.assert_array_equal(reference, retried)


class _FakeSizer:
    """Just the bookkeeping the retry touches."""

    def __init__(self, cum_rows):
        self.cum_rows = cum_rows
        self.max_workable_rows = None
        self.trace_label = "t"

    note_memory_failure = chunk.ChunkSizer.note_memory_failure


def test_split_chunk_is_reported_to_the_sizer():
    # the sizer measures per-row cost as peak memory over the rows it believes produced that
    # peak. A split chunk breaks that: the peak belongs to one piece, the row count to the
    # whole chunk. Uncorrected, the cost reads low and the NEXT chunk is sized LARGER -- the
    # retry would feed the growth it exists to contain.
    data = pd.DataFrame({"x": range(100)})
    sizer = _FakeSizer(cum_rows=100)

    def work(chunk_df):
        if len(chunk_df) > 25:
            raise MemoryError("too big")
        return len(chunk_df)

    out = chunk.run_with_memory_retry(work, data, chunk_sizer=sizer, trace_label="t")

    assert sum(out) == 100  # all rows still processed
    # 25 rows produced the peak, not 100, so the other 75 must not be charged against it
    assert sizer.cum_rows == 25
    # and the size that worked is remembered for later chunks
    assert sizer.max_workable_rows == 25


def test_unsplit_chunk_leaves_the_sizer_alone():
    data = pd.DataFrame({"x": range(100)})
    sizer = _FakeSizer(cum_rows=100)

    chunk.run_with_memory_retry(lambda df: len(df), data, chunk_sizer=sizer)

    assert sizer.cum_rows == 100  # nothing was split, nothing to correct
    assert sizer.max_workable_rows is None


def test_sizer_holds_later_chunks_at_the_workable_size():
    # once recovery has established a workable size, the sizer must not propose more --
    # the budget and the per-process memory cap are different limits and only recovery sees
    # the second one
    sizer = _FakeSizer(cum_rows=1000)
    sizer.note_memory_failure(proposed_rows=800, workable_rows=100)
    assert sizer.max_workable_rows == 100
    # a later, smaller finding tightens it further; a larger one does not loosen it
    sizer.note_memory_failure(proposed_rows=100, workable_rows=50)
    assert sizer.max_workable_rows == 50
    sizer.note_memory_failure(proposed_rows=400, workable_rows=200)
    assert sizer.max_workable_rows == 50


def test_reporting_failure_never_breaks_the_run():
    # bookkeeping must not be able to take down the work it is protecting
    class _Broken:
        def note_memory_failure(self, *a, **k):
            raise RuntimeError("sizer is unhappy")

    data = pd.DataFrame({"x": range(40)})

    def work(chunk_df):
        if len(chunk_df) > 20:
            raise MemoryError("too big")
        return len(chunk_df)

    out = chunk.run_with_memory_retry(
        work, data, chunk_sizer=_Broken(), trace_label="t"
    )
    assert sum(out) == 40


def test_retry_without_state_still_runs():
    # state is optional: callers whose work draws no random numbers need not supply it
    persons = _persons(4)
    out = chunk.run_with_memory_retry(lambda df: len(df), persons)
    assert out == [4]


def test_run_with_memory_retry_alts_exhaustion_reraises():
    choosers, alts = _choosers_and_sampled_alts(8)

    def always_fails(chunk_df, alt_df):
        raise MemoryError("always")

    with pytest.raises(MemoryError):
        chunk.run_with_memory_retry_alts(always_fails, choosers, alts, trace_label="t")
