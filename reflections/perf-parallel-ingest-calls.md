# Perf: Parallel concept/entity LLM calls in ingest pipeline

**Branch:** `perf/parallel-ingest-calls`  
**Issue:** #13  
**Date:** 2026-06-05  
**Tests:** 442 passing · 96% statement coverage

---

## What Changed

Stage 2 (concept calls) and Stage 3 (entity calls) in `_run_llm_pipeline` now run in
parallel using `ThreadPoolExecutor`. Each stage fans out all calls concurrently, then
the main thread writes results to disk + DB as futures complete via `as_completed`.

**Before:** 11 concepts × ~5s per call = ~55s sequential  
**After:** max(all 11 concept calls) ≈ ~5s wall-clock

---

## Design Decisions

**Workers do LLM call + parse only.** SQLite connections are not safe to use from
multiple threads. Rather than passing `check_same_thread=False` or adding a write
lock, workers return `(slug, parsed_dict, index)` and the main thread handles all
disk writes and DB upserts. This keeps the connection on its origin thread with
zero concurrency risk.

**Checkpoints still fire as each call completes.** `as_completed` in the main
thread processes futures as they finish, so `✓ adam-optimizer (3/11)` prints
immediately after that call returns — not in a batch at the end. Same UX as before.

**Stage ordering preserved.** `ThreadPoolExecutor` for Stage 2 is entered as a
context manager — it blocks until all concept futures complete before Stage 3
begins. Entities always start after all concepts are written and `schema` is
refreshed, matching the original sequential guarantee.

**`max_workers=min(N, 8)`** — caps at 8 threads regardless of concept/entity count
to avoid hammering the API with huge bursts on very dense sources.

---

## Live test results

**CLI confirmed working.** `medulla ingest Adam-Optimizer.pdf` with 10 concepts + 7 entities: checkpoints arrived out of order (e.g. `✓ rmsprop (5/11)` before `✓ adagrad (3/11)`), confirming concurrent execution. Measurable wall-clock speedup vs. prior sequential runs.

**MCP parallelism unclear.** The `medulla_ingest` tool description was updated to instruct Claude to batch all concept calls in one response and all entity calls in one response. In practice, Claude partially followed this — first two calls were sequential, then batched 5 in one turn. The benefit is fewer Claude round-trips (1 turn for 5 stores vs. 5 turns), not concurrent server execution. The MCP stdio server processes requests sequentially regardless of batching. True concurrent MCP execution requires HTTP transport (`medulla mcp --http`), tracked as a future issue.

## Thread safety note

The only shared mutable state during parallel execution is the `provider.generate`
call itself. All three provider implementations (Bedrock, Anthropic, Ollama) create
their API client objects inside `generate()` and are therefore stateless across
calls — safe to call concurrently from multiple threads.

---

## Tests Added

`tests/test_ingest_pipeline.py` — `MultiConceptProvider` (3 concepts, 2 entities,
records thread IDs and call times):

- `test_parallel_all_concept_pages_created` — all 3 concept files exist after ingest
- `test_parallel_all_entity_pages_created` — all 2 entity files exist after ingest
- `test_parallel_all_pages_in_db` — all 5 slugs present in DB
- `test_parallel_concepts_use_multiple_threads` — concept calls run on >1 thread
- `test_parallel_entities_start_after_all_concepts` — no entity call starts before last concept
- `test_parallel_faster_than_sequential` — total time < 3× single-call sleep
