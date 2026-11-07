# Readability follow-up

This is a second set of at most five rounds, authorized on 2026-11-07. It starts
from the first review's 8.76/10 working tree. Earlier uncommitted edits are
preserved. The target is at least 9.5 using the unchanged criteria and weights
in [the original rubric](readability-review.md#what-earns-1010).

The new request permits the focused refactors needed to address the earlier
deductions. It does not turn the proposed product in `plan.md` into an already
implemented release. Public behavior, numerical rules, state formats, errors,
and existing unsupported-feature boundaries must remain intact. No feature
removal, dependency change, public API redesign, or speculative framework is
part of this work.

## Starting assessment

| Aspect | Weight | Start | Concrete work needed for 10/10 |
| --- | ---: | ---: | --- |
| Intent and naming | 12% | 8.9 | Give recursive codec operations clear encoding/decoding names and clarify ambiguous lifecycle operations. |
| Control flow | 14% | 8.6 | Separate scheduler resolution decisions and make the iteration lifecycle and coordinator dispatch readable in coherent steps. |
| Minimality | 12% | 9.0 | Inspect a scheduler callable once; remove identical mode-restoration logic and any proven redundant work. |
| Cohesion and navigation | 10% | 8.6 | Put journal queries behind journal methods, keep endpoint dispatch out of payload handling, and give iteration work a clear owner. |
| Abstraction and indirection | 10% | 9.0 | Clarify codec direction and meaningful operation boundaries without pass-through wrappers or a generic serialization framework. |
| Duplication and consistency | 8% | 8.8 | Share mixed train/eval flag restoration while preserving differing evaluation/autograd rules. Keep unrelated validation contracts separate. |
| State and side effects | 10% | 8.6 | State the engine's boundary and restore invariants, distinguish iteration effects from lifecycle orchestration, and preserve I/O and mutation order. |
| Contracts, types, and errors | 8% | 8.5 | Give stable internal snapshot/counter records structural types and explain owner-level contracts without a second schema or changed validation. |
| Comments and documentation | 8% | 9.0 | Explain restore/report guarantees and non-obvious mutation boundaries at their owner. Remove only proven repetition. |
| Layout and scanability | 8% | 8.5 | Unpack crowded records and compound conditions in the remaining reviewed modules, without mechanically expanding simple expressions. |

Scores remain judgments of the whole reviewed code, not counts of edits or
test cases. The parent assigns each new score after inspecting a round's diff
and verification. A target score does not justify changing the rubric.

## Plan compatibility and verification

`plan.md` explicitly labels its examples and milestones as proposed work.
The implemented alpha's precise boundaries are in
[alpha-support.md](alpha-support.md). The following checks protect the relevant
existing plan contracts.

| Plan sections | Implemented contract | Checks |
| --- | --- | --- |
| 4, 15, 16 | Common training styles, synchronous named updates, weighting, clipping, successful-update schedules, evaluation | CPU PyTorch references and focused lifecycle checks. |
| 6, 7, 15.3, 18 | Deferred construction/planning, placement/aliases, immutable data descriptions, named-axis autograd | Existing focused checks plus transport/data/model regression checks. |
| 9, 13, 19, 20 | Authenticated enrollment, roles, durable single-worker submission, source archives, events and reconnection | Real localhost coordinator and worker, including environment preparation and common engine execution. |
| 10, 16, 21 | Complete local checkpoints, compatible restore, safe update boundaries and integrity | Exact interrupted/uninterrupted model, optimizer, schedule and RNG comparison; corruption and boundary checks. |
| 11 | DQN collection, replay, target updates and environment cleanup | Existing deterministic DQN baseline, including unchanged weights and counters. |
| 12, 23 | Local inference export, verified load and prediction | Matching predictions after local and worker-side export. |
| 8, 9, 10, 12 | Deferred distributed/cloud/S3-checkpoint/remote-inference capabilities | Explicit unsupported errors remain. These are not functionality claims. |

Tests added for this pass use standard-library `unittest` and the existing
PyTorch runtime. The combined suite also includes pytest cases. Install
`.[test]` in an isolated environment, then run `python -m pytest -q` from the
repository root. The real worker check and pre-refactor numerical baselines
also run from temporary scripts. No GPU, cross-machine Linux/CUDA, TLS
deployment, or S3 credentials are available for those integration paths;
their validation limits remain explicit.

## Rounds

The first round will clarify scheduler setup and consolidate evaluation-mode
restoration. Later rounds will address journal/route ownership, codec intent,
engine contracts/lifecycle, and remaining layout issues. Each implementation
agent uses GPT-6.1 Sol at high reasoning effort and receives the latest scores,
remaining deductions, preservation requirements, and regression commands.

### Round 1

Scheduler construction now names the builtin-horizon, custom-source,
multiplier-signature, and native-validation phases. Callable inspection runs
once. Learning-rate formulas and successful-update advancement are unchanged.
A shared internal context manager owns mixed train/eval flag restoration for
the engine and DQN, while each caller retains its original autograd context.

Parent review accepted 156 changed production lines across four existing files.
The focused training checks, DQN baseline comparison, compilation and diff
checks passed. New scheduler cases cover signature inspection, native rejection
precedence and restoration. The unchanged rate formulas were also compared
as syntax trees. The full localhost workflow passed before these edits.

Overall after round 1: **9.03/10**. Current aspect scores in rubric order are
9.0, 9.0, 9.4, 8.9, 9.1, 9.4, 8.9, 8.6, 9.1, and 8.8. The main remaining
deductions are journal/route ownership, codec intent, engine state contracts
and iteration flow, plus layout in the remaining dense records.

### Round 2

The coordinator now dispatches archive, worker and run requests to three
cohesive handlers. The journal owns archive-assignment authorization, status
counts and worker departure, including their original locks and transaction.
Report docstrings state input, ownership, rollback, terminal-state and duplicate
delivery guarantees at the owner.

Parent review accepted 204 changed production lines in two existing modules.
The original loopback cases and 16 added malformed/unauthorized HTTP cases
passed. Cluster smoke results match baseline; SQL strings, existing signatures
and non-routing logic were also checked as syntax trees. Authorization, body
parsing, identifier validation and error-response precedence are preserved.

Overall after round 2: **9.20/10**. Current aspect scores in rubric order are
9.0, 9.2, 9.4, 9.3, 9.2, 9.4, 9.1, 9.0, 9.4, and 9.0. Codec naming and
traversal intent, snapshot/counter contracts, iteration lifecycle, and crowded
records remain the concrete deductions for subsequent rounds.

### Round 3

Codec method and child-callback names now distinguish encoding, validation and
argument conversion. Local object identities are cached and named separately
from wire recipe IDs. Comments explain cycle-before-alias and completed-recipe
ordering. Tagged records and compound guards are easier to scan. The shared
argument traversal remains because splitting it would duplicate the grammar.

Parent review accepted 200 changed production lines across three existing
files. The 12 transport tests passed before and after editing, including new
cycle/alias and state-copy cases. Compilation and diff checks passed. Normalized
syntax-tree comparisons account only for renames, cached identities and the
new docstring. The parent tightened that docstring to distinguish local inputs
from tagged decoding inputs; no codec behavior changed.

Overall after round 3: **9.36/10**. Current aspect scores in rubric order are
9.6, 9.3, 9.5, 9.3, 9.5, 9.4, 9.2, 9.1, 9.4, and 9.2. The remaining work is
the engine's implicit state contracts and long iteration lifecycle, followed
by selective layout cleanup. All original transport formats and public
signatures remain.

### Round 4

One `_run_iteration` method now owns the complete iteration transaction. Fit
keeps epoch/budget/resume orchestration, terminal flushing and failure cleanup.
Private structural types describe complete engine counters, data-resume data
and snapshots. Partial remote counters remain permissive. Engine references
use type-checking imports without eager import cycles. Owner docstrings state
safe boundaries and explicitly preserve non-atomic restore semantics.

Parent review accepted 139 changed production lines across three existing
files. The 14 training contract tests and all temporary smoke groups passed
before and after editing. The full 30-test set also passed. Inlining the new
iteration method reproduces the complete previous fit syntax tree. Snapshot
field expressions and evaluation order match; all 24 other existing engine
methods and existing run/type method bodies are unchanged apart from docs
and typing.

Overall after round 4: **9.49/10**. Current aspect scores in rubric order are
9.6, 9.5, 9.5, 9.5, 9.5, 9.4, 9.5, 9.5, 9.6, and 9.2. Layout retains its
previous score because crowded records and compound conditions remain in
planning, data and artifact validation. The final round will address those
expressions without changing executable syntax trees.

### Round 5

Dense planning records, manifest guards, inference/checkpoint metadata, data
schema checks, loss arguments and worker subprocess/report records now expose
their fields and conditions on separate lines. The change is selective, with
293 changed production lines across eight existing modules. The parent's
strict syntax-tree comparison confirms exact equality in all eight files,
including literals, annotations, argument/key order and context managers.
No runtime operation or public contract changed in this round.

The agent's 30 regression cases passed. The parent has also made the baseline
localhost worker workflow a durable integration test, included in the normal
test command. Source distributions include the tests and review documents;
the dependency list and wheel's production-module set remain unchanged.

Overall after round 5: **9.52/10**. Only layout increases in this round, from
9.2 to 9.6; every other aspect score is retained. This completes the five-round
limit for the follow-up.

## Final scores

| Aspect | Start | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intent and naming | 8.9 | 9.0 | 9.0 | 9.6 | 9.6 | 9.6 |
| Control flow | 8.6 | 9.0 | 9.2 | 9.3 | 9.5 | 9.5 |
| Minimality | 9.0 | 9.4 | 9.4 | 9.5 | 9.5 | 9.5 |
| Cohesion and navigation | 8.6 | 8.9 | 9.3 | 9.3 | 9.5 | 9.5 |
| Abstraction and indirection | 9.0 | 9.1 | 9.2 | 9.5 | 9.5 | 9.5 |
| Duplication and consistency | 8.8 | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 |
| State and side effects | 8.6 | 8.9 | 9.1 | 9.2 | 9.5 | 9.5 |
| Contracts, types, and errors | 8.5 | 8.6 | 9.0 | 9.1 | 9.5 | 9.5 |
| Comments and documentation | 9.0 | 9.1 | 9.4 | 9.4 | 9.6 | 9.6 |
| Layout and scanability | 8.5 | 8.8 | 9.0 | 9.2 | 9.2 | 9.6 |
| **Weighted overall** | **8.76** | **9.03** | **9.20** | **9.36** | **9.49** | **9.52** |

The final score is **9.52/10**, up from **8.76/10**. The criteria and weights
are unchanged. This follow-up changes 19 existing production files, with 673
additions and 320 deletions relative to the preserved starting tree. Moved
code counts on both sides; many additions are record layout, explicit types
or contract documentation. No production module or runtime dependency was added.

Four focused regression modules are now reusable from the repository. Tests
protect behavior rather than private helper names. A concurrently prepared
contribution workflow adds further pytest cases; those files and their test
configuration were preserved and included in final verification.

## Why the score is still below 10

The remaining costs are small and specific:

- Worker monitoring still combines liveness reconciliation, cancellation and
  report batching. Its mutation order is preserved and tested, but reading the
  entire lifecycle requires following several persisted records.
- Some stable worker/assignment records remain plain dictionaries. Their
  ownership is clearer, but a future structural type could reduce key tracing.
- Small length probes are duplicated across data and training. Consolidation
  needs a useful common owner rather than another shallow utility module.
- The builtin-loss adapter retains a zero-denominator fallback dominated by
  earlier reduction guards. Removing it would be a separate behavior-checked
  simplification.
- Some archive and environment guards remain crowded, and planner inventory
  normalization could document its accepted compatibility aliases locally.

These findings do not justify weakening validation, deleting supported inputs,
or introducing a generic framework. Five rounds were used, and the requested
target was reached without claiming a perfect codebase.

## Final verification

The combined working tree passed **39 pytest tests** in 81.73 seconds, including
the focused contract cases and the concurrently added contribution cases.
Earlier standalone unittest checks also passed. Important evidence includes:

- Three training styles match raw PyTorch parameter updates and optimizer state.
- Unequal accumulation, short final windows, zero-weight skips and named updates
  preserve the mathematical objective and update ordering.
- Interrupted/resumed training matches uninterrupted weights, optimizer,
  scheduler, counters and Torch RNG behavior.
- Stop, cancellation, double-update failure and checkpoint boundaries retain
  their prior event/counter/gradient ordering.
- Codec cycles, aliases, tags, error paths and worker-relative paths are preserved.
- A real localhost worker prepares its job environment, trains, reconnects,
  copies checkpoints, drains and exports matching predictions. Other contribution
  cases verify a failed remote job is terminal and the worker accepts a successor.
- All **167 exported API signatures**, the export list and version match the
  preserved pre-pass tree. All 54 implementation submodules import.
- All original smoke outputs match baseline, both CPU examples run, and CLI
  help/version checks pass.
- The final eight-file layout round has exact AST equality. Inlining the
  iteration helper reconstructs the previous complete fit AST; optimizer math
  and snapshot expression order are unchanged.
- A 500-update trajectory has identical weights, counters and serialized-spec
  fingerprints. An initial CPU timing difference did not repeat in alternating
  before/after runs. This is a small spot check, not a throughput guarantee.
- The supplied `plan.md` remains byte-for-byte unchanged.

The default shared environment lacks pip/pytest. Verification used a disposable
environment with read-only references to the existing Torch/development
packages, plus isolated pytest tools. `pip check` passed. No production
dependencies, personal configurations, cloud resources or publication were
changed. The root environment was not updated.

The evidence is Python 3.12.1, PyTorch 2.2.2, Intel macOS and CPU/loopback
execution. It does not establish GPU/mixed-precision, TLS, S3, multi-host,
distributed-backpropagation or full dependency-range correctness. The proposed
capabilities in the product plan remain explicitly deferred where the alpha
does not implement them.

## Review scope and provenance

The task reused its existing checkout on branch `main`, based on
`030d605a8de4e2c0f9ff62732ee1239b4849a46b`. The user explicitly requested keeping
that branch and checkout, completing the tests and review, and committing and
pushing the result without a pull request. Earlier readability changes are retained.

The contribution workflow under `.agents/`, `.github/`, `AGENTS.md`,
`CONTRIBUTING.md`, the pytest fixtures and `test_training.py`,
`test_cluster.py`, `test_cli.py` appeared concurrently. They are not this
refactor's authored changes. Their configuration and execution inputs are
included in the combined-tree fingerprint and test evidence.

The parent reviewed each implementation agent's diff and ran combined checks.
A separate correctness review checked commit
`0e210e57ba4baf13c05d4c19e2154802226760b2` and found no introduced production
defects. Its one finding was a unittest discovery command that required an
uninstalled pytest dependency. The instructions now install the test extra
and use pytest for the combined suite. The final handoff identifies the
checked commit and review result. No branch
protection, hosting settings, cloud resources or merge rules are changed.

Independent review:
<t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A03d6cdcc-bd99-47a1-ae41-e87df8038f34>.
It ran 32 focused source tests and verified the CI action pins and package
contents. The parent self-reviewed the documentation-only correction and ran
the newly documented editable installation, the complete suite (**39 passed
in 73.54 seconds**) and `pip check`. Production code, tests and package
configuration are unchanged from the independently reviewed commit.

Each round used GPT-6.1 Sol at high reasoning effort:

| Round | Agent run |
| --- | --- |
| 1 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3Acbef456a-a9c7-4bf0-b04c-d560b3b5998d> |
| 2 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3Acfc20258-730e-45d8-b776-33f2fe398746> |
| 3 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3Affe9587a-547e-40de-948d-d5104fdde121> |
| 4 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A8fa4a877-b2f9-45c9-be19-cf1ddfa046c7> |
| 5 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A2b5b6061-4345-403e-b4ef-d39573b84d96> |

The [score chart](readability-follow-up-scores.html) uses these same totals.
Package build and installed-wheel checks are recorded below.

### Package and installed-wheel checks

The wheel and source archive built successfully with `python -m build
--no-isolation`. Both passed `python -m twine check`. The wheel contains every
production Python module, and the source archive includes the regression tests
and review documents.

The wheel was installed without dependencies into the disposable verification
environment. From outside the checkout, isolated import/version matching,
`python -I -m mlzo --help`, the installed `mlzo --json --version` entrypoint,
and all 14 training contract tests passed. The import path was checked to
ensure the installed wheel, rather than editable source, supplied the package.
The installed environment also passed `pip check`.

Final compilation covers all 70 Python files now present in the combined
checkout. The contribution workflow and test additions were preserved; no
GitHub repository settings or cloud resources were changed.
