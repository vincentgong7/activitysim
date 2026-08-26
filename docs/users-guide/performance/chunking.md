# Chunking

The default operation of ActivitySim is to attempt to run simulations in each
component for that component's entire pool of choosers in a single operation.
This allows for efficient use of vectorization to speed up computations, but can
also lead to memory issues if the pool of choosers is too large.  This is particularly
a problem in interaction-type models, where a large pool of choosers is faced
with a large set of alternatives.

ActivitySim includes the ability to "chunk" these model components into more
manageable sized groups of choosers, which can be processed one chunk at a time.
There is a small overhead associated with chunking, but if the total number of
chunks is relatively small, the overhead is usually outweighed by the benefits
in reduced memory usage.

Chunking can be used in two ways in ActivitySim: dynamic and explicit.  Dynamic
chunking is the original chunking system in ActivitySim, and it remains available
to support users already familiar with it.  It is designed to strive for optimal
chunk sizes, but is complicated to use. Explicit chunking is simpler to use
and understand, and is recommended for most users.

## Dynamic Chunking

This is the original chunking system in ActivitySim, where model components are
chunked into pieces that are selected to be approximately optimal for targeting
a particular memory usage threshold.  The chunk size is determined by "training"
the model so that it can estimate the memory usage to simulate each chooser handled
in each component, and then running in "production" mode where the chunk size is
set to keep the memory usage below the selected threshold, based on the results from
the training.

To configure chunking behavior, ActivitySim must first be trained with the model
setup and machine.  To do so, first run the model with ``chunk_training_mode: training``.
This tracks the amount of memory used by each table by submodel and writes the results
to a cache file that is then re-used for production runs.  This training mode is
*significantly* slower than production mode since it does a lot of memory inspection.
For a training mode run, set ``num_processors`` to about 80% of the available logical
processors and ``chunk_size``to about 80% of the available RAM.  This will run the
model and create the ``chunk_cache.csv`` file in the cache directory for reuse.  After
creating the chunk cache file, the model can be run with ``chunk_training_mode: production``
and the desired ``num_processors`` and ``chunk_size``.  The model will read the chunk
cache file from the cache folder, similar to how it reads cached skims if specified.
The software trains on the size of problem so the cache file can be re-used and
only needs to be updated due to significant revisions in input file or changes in
machine specs.  If run in production mode and no cache file is found then ActivitySim falls
back to training mode.

For more detail on running with dynamic chunking, see [Chunking](chunk_in_detail).

## Explicit Chunking

This is a simpler system that allows the user to specify the number of choosers
in each chunk explicitly, either as an integer number of choosers per chunk, or
as a fraction of the overall number of choosers. Although the total amount of
memory engaged for processing any particular chunk is ignored and there is no
effort to find a "optimal" chunk size, this system is easier to use
and understand than dynamic chunking, and in practice has been found to be more
robust and reliable. It requires no "training" and is activated by setting the
`chunk_training_mode` configuration setting to `explicit`.

This method for chunking does rely upon model developers to have identified the
memory-hungry components and to have set reasonable explicit chunk sizes for them.
See [this notebook](https://github.com/ActivitySim/activitysim/blob/main/other_resources/scripts/plot_memory_profile.ipynb)
for an example of how to review component memory usage.
Individual model components are configured to use chunking explicitly by
setting the `explicit_chunk` configuration setting in the model component's
settings, when available. (Refer to each model component's documentation for
details on whether explicit chunking is available with that component.)  The
chunk setting can be set to an integer number of choosers to process in each
chunk, or to a fractional value to make chunks approximately that fraction of
the overall number of chooser (e.g. set to 0.25 to get four chunks).

## Recovering From Running Out of Memory

Every chunking strategy on this page is a prediction, and each of them has the same
weakness: the prediction has to be right the first time, because being wrong ends the
run.  Inside a container that ending is abrupt.  The kernel kills every process in the
cgroup together, so nothing gets the chance to report what happened — the parent never
sees a worker die, and every log stops in the same second without recording which model
was responsible.

Setting `memory_fail_recovery: true` (Linux only, off by default) changes that.  Each
worker caps its own anonymous memory just below what the container can give it, so an
over-large allocation raises a catchable `MemoryError` in the worker that asked for it
rather than pushing the container over its limit.  The chunk loop then halves that chunk
and runs it again, and tells the chunk sizer what size did fit so later chunks start
from there.

The intent is not to make a badly-sized run fast.  It is to stop a badly-sized run from
being fatal:

* where the memory is being used *inside* a chunk loop, a wrong chunk size costs a
  retry rather than the run;
* where it is being used *outside* one — building a chooser table, joining sampled
  alternatives, writing summaries — nothing can be split, so the run still fails, but it
  fails with a traceback naming the step and the allocation instead of vanishing.

Two properties are worth knowing before enabling it.

**Results do not change.** Each row's random draws are seeded from its own index, so
splitting a chunk is invisible to the random streams, and a retry rewinds the stream
position of the rows it re-runs.  A run that hits retries produces the same choices as
one that does not.

**It is not a licence to under-provision.** A cap cannot conjure memory that is not
there.  If the ceiling is below what a component needs before it even reaches chunked
work, the run fails immediately — cleanly and legibly, but immediately.  Relaxing
`chunk_size_safety_factor` because retries will catch the overruns is also a poor trade:
it buys little time and spends most of the headroom that keeps a run away from the
ceiling.

Two settings control it:

* `memory_fail_recovery` — off by default; the whole mechanism.
* `memory_step_cap_ratio` — the fraction of still-available memory a whole model step
  may grow into, default `1.0`.  At the default this ceiling sits at the container limit,
  where it can only catch allocations that were going to fail anyway.  Lower it to be
  told sooner, at the risk of stopping a step short of memory it could have used.
