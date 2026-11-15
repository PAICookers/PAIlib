# PAIlib Agent Guide

This file records guidance for agents working in the PAIlib repository.

## Operating Rules

- Treat this checkout as user-owned. Do not revert existing dirty changes unless
  the user explicitly asks.
- For non-trivial work, write the workspace decision, scope, checkable plan,
  verification, review, and remaining risk in `tasks/todo.md`.
- Keep durable project lessons in `tasks/lessons.md` when the user corrects a
  repeatable mistake pattern.
- Use a dedicated worktree only for broad, long-lived, parallel, or collision-prone
  tasks. Narrow documentation or single-module edits can stay in this checkout.
- Prefer codebase-memory for structural discovery, following the on-demand
  procedure below. Use `rg` for literals/configuration or index gaps and
  `git ls-files` for tracked paths. When a tracked file has been
  renamed in the worktree but not yet staged, inspect both the deleted tracked
  path and the new path before treating the new path as unrelated.
- Before closing documentation-only work, at minimum run whitespace/diff checks.
  For Python source or test changes, run focused pytest, `ruff check`,
  `ruff format --check`, and `git diff --check` on the touched paths.
- Do not broaden scope into sibling repositories unless the user names them.

## Project Purpose

`paicorelib` is the Python support library for PAICORE and PAICORE 2.5. It models
chip coordinates and routing, defines core/neuron register contracts, validates
parameters with Pydantic, generates hardware frame packages, packs dense/CSC
weights, and encodes BF16/FP32 payload bits.

The public package is exported through `src/paicorelib/__init__.py`. The project
targets Python 3.10+ (CI currently covers 3.10-3.14) and uses NumPy, Pydantic v2,
uv, pytest, Ruff, and pre-commit. `pyproject.toml`, `uv.lock`, `ruff.toml`, and
`.pre-commit-config.yaml` are the tooling sources of truth.

## Code Style and Contracts

- Use clear `snake_case` functions/variables, `PascalCase` classes, and
  `UPPER_CASE` constants. Follow nearby code and keep public names explicit.
- Ruff is the formatter and import/lint checker; use the repository config.
  Do not add a second formatter or rewrite unrelated files to change style.
- Do not introduce `from __future__ import annotations`. Keep annotations valid
  on every supported Python version; quote references imported only under
  `TYPE_CHECKING`. After annotation changes, smoke-import the affected modules
  on the minimum and maximum CI Python versions.
- Use typed public APIs and `numpy.typing.NDArray` with explicit dtype contracts
  for arrays. Choose integer dtype, signedness, masks, and intermediate widths
  from hardware semantics; test boundaries and overflow before narrowing values.
- Preserve strict Pydantic/enum validation. Keep `typing_extensions.TypedDict`
  for Pydantic contracts while supporting Python 3.10/3.11. Keep BF16/FP32
  numeric carriers separate from hardware bit reinterpretation.
- Search callers, exports, fixtures, and tests before changing a contract.
  Reuse existing helpers; avoid no-op bases, one-use abstraction layers, and
  duplicated state. Preserve error handling when moving or combining code.
- Library code must not print or configure global logging as an import side
  effect. Follow existing module logging patterns and explain only non-obvious
  hardware constraints in comments.
- Keep hardware facts, software design, test results, and open questions
  distinct. Use repository-relative paths in shared documentation; never add
  credentials, raw private dumps, or machine-specific paths to shared rules.

## Codebase Memory: Read Only What the Task Needs

The graph is a navigation aid, not proof of runtime behavior. Do not preload
the whole graph, every module, or unrelated project indexes at session start.

1. Check `git status --short --branch` and the checkout root. Before the first
   structural lookup (and again after compaction or a checkout change), call
   `index_status(project="@current")` once and inspect root, freshness, parse
   gaps, and watch state. If this server does not support `@current`, discover
   the exact project with `list_projects` or the returned project list, then
   confirm its `root_path` is this checkout using `index_status`. Never guess
   a project name from a branch or reuse another checkout's graph silently.
2. Select the smallest lookup for the question:

   | Need                                          | Read on demand                                                                                        |
   | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
   | Known symbol/module                           | `search_graph` with name and file filters, then `get_code_snippet` for the relevant definition    |
   | Caller or contract impact                     | Locate the exact symbol, then `trace_path` inbound; add outbound/both only when dependencies matter |
   | Cross-module relationship                     | A bounded `query_graph` with an explicit `LIMIT` after the symbols/edges are known                |
   | Unfamiliar overall architecture               | One `get_architecture` call, then narrow to the owning module                                       |
   | Literal, YAML, tooling config, missing source | Scoped `search_code` or `rg`, then read only the matching source ranges                           |

   Start with a small result limit (for example 10) and narrow paths. Follow
   `has_more`/pagination only when the remaining results can change the answer.
   Stop once the owner, relevant callers, contract, and verification path are
   established. Read whole files only when the change spans their contents.
3. Before relying on graph findings, call `check_index_coverage` once for the
   evidence paths. For absence/exhaustiveness claims include the relevant
   scopes and finish their pagination. Read current source for skipped,
   partial, excluded, stale, or unknown ranges. An empty result never proves
   code is absent; successful coverage is still best-effort evidence.
4. If `pi_cbmem_resolution` reports `borrowed_canonical_base`, keep `@current`
   for supported graph reads. Use the reported source project only for the
   coverage check and then revalidate `@current`. Do not bypass a dirty or
   divergent worktree rejection by selecting that canonical project directly;
   index this checkout or fall back to current source.
5. Treat `@current` as a read alias, not an index identity. Do not pass it to
   `index_repository`, `search_code`, `check_index_coverage`, or
   `detect_changes`; use the validated exact project where required.
6. If delegating research, pass only the bounded scope, project/root/freshness,
   symbols/paths, coverage gaps, and open questions. A recipient without graph
   tools must use the supplied evidence and current source, not claim graph use.

### Refresh Only When Needed

- Refresh after structural source/test changes (symbols, signatures, imports,
  exports, call relationships, additions, deletions, or renames), a stale or
  missing index, branch changes that invalidate the snapshot, or an explicit
  user request. Inspect `watch`: do not assume this checkout is auto-watched.
- For comment/docs/config-only edits, read those files directly; do not rebuild
  an otherwise current code graph solely because a task began or ended.
- Refresh the actual checkout with
  `index_repository(repo_path="<absolute checkout root>", mode="full", persistence=true)`.
  Use `async=true` for a longer run; poll the same root with `status=true`
  until it succeeds or fails. Do not report queued/running work as complete.
- After refresh, check `index_status` and coverage for touched/relevant source
  paths, inspect parse gaps, and smoke-query one affected symbol. Record the
  verified commit/branch and material exclusions; do not claim this graph
  indexes only `src/` and `tests/` unless coverage confirms that scope.
- `.codebase-memory/graph.db.zst` is the persisted graph; inspect generated
  metadata for freshness, not as a substitute for source or tests.
  **Keep `.codebase-memory/artifact.json` untracked and unignored.** Never add
  ignore/exclude rules, skip-worktree, or assume-unchanged flags to hide it.
  If staging deliverables, name files explicitly and check the staged paths;
  do not stage the entire `.codebase-memory/` directory.

## Architecture Map

- `src/paicorelib/hw_defs.py`: old and v2.5 chip constants, coordinate limits,
  offline/online core layout, and hardware parameter groups.
- `src/paicorelib/coordinate.py`: old `Coord`/offline/online coordinates plus
  v2.5 `CoordXY`, `CoordXYOffset`, `CoordZXYOffset`, and sign-magnitude helpers.
- `src/paicorelib/routing_defs.py`: old multicast routing directions, paths, and
  replication-id expansion.
- `src/paicorelib/routing_hexa.py`: PAICORE 2.5 AER packet route/copy handling,
  hex-grid walking, area expansion, and shortest-path helpers.
- `src/paicorelib/core_defs.py` and `src/paicorelib/neuron_defs.py`: legacy core
  and neuron enums, modes, and limits.
- `src/paicorelib/core_defs_v2.py` and `src/paicorelib/neuron_defs_v2.py`:
  PAICORE 2.5 protocol enums and hardware limits for offline and online cores.
- `src/paicorelib/core_model.py` and `src/paicorelib/neuron_model.py`: legacy
  Pydantic parameter models and validators.
- `src/paicorelib/core_model_v2.py` and `src/paicorelib/neuron_model_v2.py`:
  PAICORE 2.5 strict Pydantic contracts. Keep shared common fields separate from
  protocol-specific offline/online fields.
- `src/paicorelib/float_codec.py`: BF16/FP32 parameter carriers and frame payload
  bit packing. Keep numeric carriers separate from raw hardware payload bits.
- `src/paicorelib/framelib/frame_defs.py`: frame bit-layout constants and masks
  for legacy and v2.5 frames.
- `src/paicorelib/framelib/base.py`: generic frame/package base classes and
  destination decoding helpers.
- `src/paicorelib/framelib/frames.py` and `src/paicorelib/framelib/frame_gen.py`:
  legacy frame classes and generators.
- `src/paicorelib/framelib/frame_gen_v2.py`: main PAICORE 2.5 frame generator for
  config, neuron, weight, work, control, test, dense, CSC, and static runtime
  routes.
- `tests/`: behavior reference mirroring the package layers. The v2.5 frame-gen
  tests are the strongest executable specification for frame layout, timing, LCN,
  dtype, truncation, and payload semantics.

## Verification Entry Points

Use the smallest verification set that proves the touched contract.

- Core/neuron model changes:
  `PYTHONPATH=src pytest tests/test_core_model_v2.py tests/test_neuron_model_v2.py -q`
- V2 frame generation changes:
  `PYTHONPATH=src pytest tests/framelib/test_frame_gen_v2.py -q`
- Float payload changes:
  `PYTHONPATH=src pytest tests/test_float_codec.py tests/framelib/test_frame_gen_v2.py -q`
- Routing/coordinate changes:
  `PYTHONPATH=src pytest tests/test_coordinate.py tests/test_routing_defs.py tests/test_routing_hexa.py -q`
- Formatting/static checks for touched Python files:
  `ruff check <paths>` and `ruff format --check <paths>`
- Always finish with:
  `git diff --check -- <touched paths>`
