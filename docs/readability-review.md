# Readability rubric and review

Reviewed on 2026-11-07.

This document records the first review. The [follow-up review](readability-follow-up.md)
contains the subsequent five rounds and the current 9.52/10 assessment.

This review covers the 61 tracked Python files in `src/mlzo` and `examples`,
9,557 lines at commit `030d605`. Every Python file was inspected. Package
documentation and `pyproject.toml` supply context for the supported alpha
behavior. Generated distributions, the virtual environment, and the original
product plan are outside the code scores.

The goal is to reduce the effort needed to read, review, and change the code.
Scores are editorial judgments supported by examples, not measured correctness
or a numerical certification. Necessary PyTorch, filesystem, and process rules
do not count as avoidable complexity.

## Scoring rules

Each aspect has a score from 0 to 10. The overall score is
`sum(aspect score * weight) / 100`, rounded to two decimals. Use the same
criteria and weights in every round. Retain a score when a change does not
address its evidence. Do not raise an untouched package's score by implication.

| Score | Meaning |
| --- | --- |
| 10 | The aspect meets its criteria throughout the reviewed code. No concrete, useful readability correction remains. |
| 9 | Easy to review, with a few small, local improvements remaining. |
| 8 | Generally clear, with recurring friction or a few dense sections. |
| 7 | Understandable, but reviewers repeatedly need to decode expressions or trace related behavior elsewhere. |
| 5-6 | Common changes require substantial reconstruction of intent or control flow. |
| 3-4 | Intent and ownership are often unclear, with extensive incidental complexity. |
| 0-2 | The aspect consistently obstructs review and maintenance. |

Intermediate scores express judgment within a band. Differences of a tenth
should not be interpreted as precise measurements. This is a whole-codebase
assessment based on the prevalence and severity of findings, not an average
of file lengths or a score for only the edited files.

## What earns 10/10

| Aspect | Weight | Criteria for 10/10 |
| --- | ---: | --- |
| Intent and naming | 12% | Names identify the domain concept, action, and units where relevant. Validation, conversion, querying, and mutation are distinguishable. A reader can understand a local expression without guessing an abbreviation. Conventional mathematical and PyTorch names remain appropriate. |
| Control flow | 14% | The normal path reads in execution order. Conditions and failure paths are explicit. Branches express distinct rules, with no redundant branches, nested conditional puzzles, or exception tricks. Extraction is justified only when it makes the flow easier to follow. |
| Minimality | 12% | Every statement, option, function, and module has a current purpose. There are no dead paths, needless temporary values, repetitive work, or speculative generalization. Brevity never hides validation, cleanup, or a domain rule. |
| Cohesion and navigation | 10% | A function performs one coherent job at a consistent level of detail. A maintainer can locate a rule and its owner without chasing forwarding layers. Modules group behavior that changes together. Long cohesive operations are allowed when their sequence is clearer inline. |
| Abstraction and indirection | 10% | Helpers hide meaningful complexity or remove a repeated rule. Public convenience methods have a clear user benefit. There are no shallow private wrappers, one-use frameworks, or abstractions requiring more explanation than their direct implementation. |
| Duplication and consistency | 8% | Each identical policy has one owner. Similar operations use consistent terms and patterns. Repetition is retained when rules differ in accepted values, error handling, or lifecycle. A shared helper does not erase those differences. |
| State and side effects | 10% | State ownership, mutations, I/O, resource lifetime, and ordering constraints are visible. Derived state is not independently maintained without a reason. Reviewers can identify safe boundaries, cancellation, and cleanup without reconstructing hidden mutations. |
| Contracts, types, and errors | 8% | Inputs, outputs, invariants, and failure modes are understandable at each boundary. Types add useful information. Dynamic transport values may use `Any`, but unrelated structures are not casually conflated. Errors preserve actionable context and deliberate security limits. |
| Comments and documentation | 8% | Comments explain constraints or decisions the code alone cannot explain. Public contracts describe actual behavior and limits. There are no stale statements, narration of obvious syntax, or repeated prose that obscures the implementation. |
| Layout and scanability | 8% | Related steps are grouped and unrelated steps are separated. Calls, conditions, and records can be scanned without mentally splitting crowded lines. Formatting is consistent within a module. Neither minimum line count nor a universal line-length rule substitutes for judgment. |

## Baseline

| Aspect | Baseline | Evidence preventing 10/10 | What would close the gap |
| --- | ---: | --- | --- |
| Intent and naming | 8.5 | `training/run.py` names validators `_budget` and `_config`; `training/engine.py` uses `_safe` for checkpoint-boundary validation. Some dynamic codecs use broad `_value` names. | Name consequential operations by what they do. Keep short names only where their role is apparent locally. |
| Control flow | 7.5 | `cluster/_store.py:workers` has six nested conditional outcomes. `_common.py` and `coordinator.py` raise through generator expressions in JSON callbacks. `Engine.fit` repeats `_end_epoch()` in adjacent branches. | Make status precedence explicit, use an ordinary JSON rejection callback, and remove redundant branches. Simplify remaining dense lifecycle expressions locally. |
| Minimality | 8.5 | The duplicate end-of-epoch branch adds no rule. `Worker._launch_run` and `_launch_command` repeat the same persisted-launch recovery block. | Remove the redundant branch and give the identical recovery rule one owner without broad lifecycle changes. |
| Cohesion and navigation | 8.5 | Coordinator routing mixes endpoint dispatch, validation, and direct journal access. `Engine.fit` and snapshot publication have long sequences of distinct phases. | Keep phase transitions legible and local. Separate only independently understandable work with a clear payoff. Deeper router/journal changes require separate review. |
| Abstraction and indirection | 9.0 | Public factories and PyTorch re-exports are useful and mostly direct. Some private codec callbacks and generic encode/decode paths still require tracing closures and mode flags. | Clarify the remaining helpers where a smaller direct expression preserves their error-path and transport behavior. Do not remove useful public entrypoints. |
| Duplication and consistency | 8.0 | Three cluster JSON loads repeat the same nonfinite-number rejection trick. Worker launch recovery repeats a lifecycle rule. Validators across checkpoint/inference/training look similar but have different contracts. | Share the identical cluster callback and recovery rule. Preserve deliberately different validation policies. |
| State and side effects | 8.0 | Worker recovery and monitoring require tracing several persisted files. `Engine` checkpoint records and DQN cleanup pack multiple state concerns into dense expressions. | Show mutation order and lifecycle choices explicitly. Keep persistence, rollback, and device rules intact. Wider ownership changes exceed a small readability edit. |
| Contracts, types, and errors | 8.5 | Dynamic lifecycle and transport boundaries use many bare dictionaries and `Any`; callers must inspect their keys in other methods. Error types and public docstrings already explain many constraints well. | Improve local names and record layout first. Add structural types only where they remove real ambiguity without creating a parallel schema. |
| Comments and documentation | 9.0 | Checkpoint, inference, and alpha limitations are documented well. Dense implementations still make readers correlate several comments with packed operations. | Keep necessary explanations beside the operation they constrain and remove only proven repetition or staleness. Avoid adding narration to simple code. |
| Layout and scanability | 7.5 | Training records, remote request payloads, and subprocess calls frequently pack several fields or decisions onto a line. There are 205 Python lines over 119 columns, including necessary strings. | Expand dense records, calls, and conditions where it helps compare fields or follow steps. Leave simple lines and long error messages alone. |

Weighted baseline: **8.28/10**.

## Coverage and boundaries

| Area | Review evidence | Remaining concern at baseline |
| --- | --- | --- |
| API exports, version, errors, providers | Exports are explicit; transforms delegate directly to PyTorch; unsupported provisioning fails explicitly. | Crowded export lists are minor; useful compatibility entrypoints should remain. |
| Model and planning | Lazy state transitions, alias preservation, placement, device inventory, and side-effect-free planning are explicit. | Long placement conditions and metadata records need close reading. |
| Data and transforms | Immutable pipeline descriptions and ordered adapters expose their contracts; named-axis operations are direct. | `build`/`from_spec` contain many legitimate cases; several expressions are crowded. |
| Training | Run configuration, weighted losses, optimizer ownership, schedules, iteration loop, evaluation, and restore were inspected. | Dense engine records, nested selection expressions, and broad dynamic boundary types. |
| Checkpoint | Policy, validation, manifests, publication, retention, verified reads, and restricted state copies were inspected. | Long publication sequence and validation conditions; durability rules must remain visible. |
| Cluster | Client, journal, coordinator, worker, launcher, child entrypoints, reports, archives, and codecs were inspected. | Status precedence, JSON exception trick, repeated launch recovery, and long control-plane methods. |
| Inference | Recipe/state validation, immutable export, verified load, deployment, and shutdown were inspected. | Some filesystem rules resemble checkpoints but have different contracts; automatic consolidation would be risky. |
| DQN | Collection, replay, target updates, partial resume, environment contracts, and cleanup were inspected. | Dense conditional expressions and runtime state restoration. |
| CLI and examples | Command parsing, secret handling, dispatch, and all six example files were inspected. | Long parser setup is mostly a direct declaration; avoid replacing it with a framework. |

## Improvement process

Use GPT-6.1 Sol with high reasoning effort for each implementation round.
Each agent receives this rubric, current scores, concrete deductions, and a
narrow change list. The parent reviews the diff, checks behavior, and assigns
the next scores. Agents do not award their own final score.

Start with three rounds and allow up to five if small useful changes remain.
Preserve public signatures, errors, serialization formats, status precedence,
numerical operations, resource cleanup, and supported alpha limits. No module
reorganization, new dependencies, blanket formatter run, or feature work is
part of this task. Existing short public factories are not dead code merely
because they delegate.

A tiny behavior-preserving change may add lines to remove decoding effort.
Line counts are context only. Do not chase a perfect score through broad
rewrites. Retain deductions and name the remaining work when the safe small
changes have been exhausted or five rounds have completed.

## Verification and round results

Before edits, all 61 Python files compiled under Python 3.12.1. Both CPU examples
completed, including local checkpoint, export, and prediction. CLI help and JSON
version output passed. Focused temporary checks also passed for:

- Finite JSON and rejection of NaN/Infinity at local metadata, transport-response,
  and coordinator-body boundaries.
- Nine worker-status cases, including competing stop, drain, disconnect, and
  revocation flags.
- Deferred construction, total step budgets, partial-epoch checkpoint/resume,
  evaluation mode restoration, export/prediction, empty input, and invalid budgets.

Baseline outputs were saved outside the repository for comparison after edits.
This repository intentionally has no comprehensive test suite. Validation here
uses compilation, existing CPU examples, and focused temporary smoke checks,
consistent with the alpha contract. These checks do not certify GPU, TLS, S3,
or remote subprocess execution.

### Score history

| Aspect | Baseline | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intent and naming | 8.5 | 8.5 | 8.8 | 8.8 | 8.9 | 8.9 |
| Control flow | 7.5 | 7.9 | 8.3 | 8.3 | 8.5 | 8.6 |
| Minimality | 8.5 | 8.5 | 8.7 | 8.9 | 8.9 | 9.0 |
| Cohesion and navigation | 8.5 | 8.5 | 8.5 | 8.6 | 8.6 | 8.6 |
| Abstraction and indirection | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 |
| Duplication and consistency | 8.0 | 8.4 | 8.4 | 8.7 | 8.7 | 8.8 |
| State and side effects | 8.0 | 8.0 | 8.2 | 8.5 | 8.6 | 8.6 |
| Contracts, types, and errors | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 |
| Comments and documentation | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 |
| Layout and scanability | 7.5 | 7.6 | 8.0 | 8.2 | 8.3 | 8.5 |
| **Weighted overall** | **8.28** | **8.38** | **8.54** | **8.65** | **8.71** | **8.76** |

### Round 1

The agent added one ordinary `reject_json_constant` callback in
`cluster/_common.py`, reused it at the three existing call sites, and expanded
the worker-state expression in `cluster/_store.py` into ordered branches.
The diff has 20 additions and 6 deletions across three source files.

Parent review confirmed the same status precedence, JSON rejection, and
caller-specific error wrapping. Compilation, the focused cluster checks, and
`git diff --check` passed. Control flow and duplication improved; a small
layout increase reflects the explicit branches. Other scores stay unchanged
because this round did not address their evidence.

The largest remaining small fixes are the redundant epoch-ending branch,
dense training records, and repeated worker launch recovery. Those deductions
remain. No status, error text, public signature, or wire-format change was
accepted.

### Round 2

The agent renamed the run configuration/budget validators and the engine's
checkpoint-boundary validator, updating every call site. Adapter and evaluator
selection now use explicit branches. The exhausted-input path calls
`_end_epoch()` once after the existing empty-step error check. Dense counters,
checkpoint records, status arguments, and training configuration fields were
expanded where they made review harder. Two source files changed, with 83
additions and 36 deletions.

Parent review checked validator references, the empty-source distinction,
autocast scope, mode restoration, and checkpoint key order. Compilation,
baseline-comparison training checks, and `git diff --check` passed. Naming,
control flow, minimality, visible state records, and layout improved. The
other aspect scores remain unchanged. The engine still has substantial mutable
state and a long iteration lifecycle; this round did not redesign either.

Round 3 should address the identical worker launch recovery blocks. The
recovery rule must retain an existing launch, copy a recorded child PID when
present, and persist the active job without creating a new subprocess.

### Round 3

One private `_restore_launch` method now owns persisted launch recovery for
both jobs and completed-job commands. The callers keep their distinct setup
and subprocess arguments. Active-job records and spawn calls are expanded
inside those two launch methods. One source file changed, with 38 additions
and 15 deletions.

Parent review confirmed the original file-access order, existing-launch early
return, optional child PID, and active-record fields. Compilation, focused
worker checks against baseline, and `git diff --check` passed. The recovery
checks forbid spawning and cover recovery with and without `child.json`.
Minimality, cohesion, duplication, state visibility, and layout improved.
These checks do not exercise live subprocess supervision.

The overall score remains below 9. Two more small rounds are justified by
concrete remaining findings: DQN has a nested environment-cleanup choice and
compressed runtime choices; optimizer parameter-group validation and data
source classification still have local duplication or crowded branches.
Use the allowed fourth and fifth rounds for those edits, then stop.

### Round 4

DQN cleanup now shows explicitly which environment it closes. Cached
environment removal and the optional global RNG draw are separate statements.
Action-space locals use `action_count` and `action_start`. One source file
changed, with 20 additions and 9 deletions.

Parent review checked cleanup ordering, idempotence, lazy environment behavior,
seed selection, and conversion before the paired state assignment. Compilation,
deterministic DQN checks against baseline, and `git diff --check` passed. The
tiny environment produced identical metrics, counters, and model weights;
cleanup of an unopened supplied environment also passed. Naming, control flow,
state visibility, and layout improved modestly. DQN's replay/target state and
partial-environment resume design remain unchanged.

### Round 5

Optimizer binding now names the parameter-group shape once, selects the
parameter list through explicit branches, and reuses that shape decision for
the existing empty-group check. DataLoader and Dataset classification share
their identical native-source branch. Two source files changed, with 8
additions and 6 deletions.

Parent review checked that duplicate/empty validation still precedes optimizer
construction and that native source behavior is unchanged. Compilation,
parameter-group selection, four invalid optimizer configurations, Dataset
iteration, DataLoader identity, and `git diff --check` passed against baseline.
Control flow, minimality, duplication, and layout improved modestly; other
scores remain unchanged. This completes the five-round limit.

## Final assessment and remaining deductions

The final whole-codebase score is **8.76/10**, up from **8.28/10**. Nine source
files changed, with 169 additions and 72 deletions. There are still 61 Python
files, now 9,654 lines. Most additions make existing branches, state records,
or calls explicit. The changes introduce no dependencies or module hierarchy.

The score is not 10, and the assessment does not imply that unchanged code
improved. The following concrete findings remain. These are candidates for
future authorized work, not extra edits after the fifth round.

| Aspect | Final | Evidence still preventing 10/10 | Further work needed |
| --- | ---: | --- | --- |
| Intent and naming | 8.9 | Generic `_value`/`_encode` codec helpers still require reading their caller to understand direction and accepted shapes. | Give the consequential codec operations and child callbacks more specific names where that reduces tracing. |
| Control flow | 8.6 | `Schedule.__init__` interleaves native scheduler discovery, callable arity, horizon inference, and validation. Coordinator routing is still long. | Clarify one decision sequence at a time while preserving dispatch and validation order. |
| Minimality | 9.0 | Scheduler callable validation repeats `inspect.signature(source)` for the same callable. | Consider retaining one signature after checking the introspection contract; remove other repeated work only with concrete evidence. |
| Cohesion and navigation | 8.6 | `_Handler._route` queries journal SQL directly for archive access and run counts, alongside endpoint dispatch and payload validation. | Give the journal ownership of those queries in a separate, focused change with request checks. |
| Abstraction and indirection | 9.0 | Codec recursion, child closures, and encode/decode flags still require following multiple levels to understand error paths. | Clarify the local interfaces and retain error-path context. Avoid a new generic serialization framework. |
| Duplication and consistency | 8.8 | Engine evaluation and DQN's `_evaluating` both save and restore per-module train/eval flags. | Assess a shared evaluation-mode context manager while preserving intentionally different child modes and other evaluation behavior. |
| State and side effects | 8.6 | Engine fit/restore and worker monitoring require coordinating several flags and persisted records. Safe boundaries depend on that sequence. | Document state invariants locally and review smaller lifecycle units before considering ownership changes. |
| Contracts, types, and errors | 8.5 | Snapshot, worker, and result records still use broad dictionaries and `Any`; required key relationships are implicit across methods. | Add structural types only for stable reused records, with one owner and no parallel runtime schema. |
| Comments and documentation | 9.0 | `Engine.load_state_dict` and `Store.report` lack concise owner-level explanations of their required record fields and mutation guarantees. | Add short contract explanations that record these non-obvious constraints without narrating the implementation. |
| Layout and scanability | 8.5 | Planning metadata, scheduler conditions, and checkpoint manifest validation retain crowded expressions and records. | Continue selective layout edits where reviewers must compare fields or decode compound conditions. |

Some of these changes could be small; others cross module boundaries. They
remain because the requested maximum was five rounds. A higher score needs
additional evidence and review, not a relaxed rubric. Necessary alpha
validation and durability checks should remain even when they add code.

Final verification results and agent references are recorded below.

## Final verification

All checks passed after the fifth round:

- All 61 tracked Python files compiled under Python 3.12.1.
- AST comparisons confirmed that existing public function, method, and
  constructor signatures are unchanged.
- All 54 library modules imported; the callable top-level `data`, `checkpoint`,
  and `cluster` exports remained intact.
- Combined JSON/status, training/resume/inference, worker recovery, deterministic
  DQN, optimizer validation, and native-data checks matched their saved baselines.
- Both existing CPU examples completed in fresh temporary output directories.
- CLI help and JSON version commands passed, as did `git diff --check`.

The checks used the existing PyTorch 2.2.2 environment. Temporary smoke scripts
and baseline outputs live outside the repository; no test suite, environment
change, commit, or publication was added. This review does not validate live
remote subprocess execution, GPU, TLS, S3, optional Gymnasium integration, or
the declared full Python/PyTorch version range.

## Agent runs

Each implementation run used `gpt-6.1-sol` with `reasoningEffort=high` in the
shared checkout. The parent supplied the current rubric and deductions, then
reviewed the changes, checked behavior, and assigned the next scores.

| Round | Agent run |
| --- | --- |
| 1 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A5c25c52a-f0fd-4e38-a7fd-d4b4cd3fa80e> |
| 2 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A98f30d78-272b-43ab-8ed3-5458ebd27e92> |
| 3 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A7d6ac7f8-73dd-4969-bbb3-3bf957e04420> |
| 4 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A3b5c6340-d94f-4670-86f5-33b9ee7fceb5> |
| 5 | <t3-thread://v2/596701e5-5153-471d-a553-7727cb55f3f3/thread:delegated-task:command%3Amcp%3Ac169140e-ac25-44fc-a86b-8b887a4b610b%3Adelegate-task%3A26bf2b3e-6ae3-41cc-9db1-f7906c36f936> |

The [score chart](readability-scores.html) uses the same weighted totals as
the score-history table. Its full 0-10 scale avoids exaggerating the changes.
