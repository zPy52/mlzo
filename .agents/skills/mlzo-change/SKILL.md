---
name: mlzo-change
description: Implement and verify a focused MLZO change in its own worktree, obtain local review, and prepare evidence for a human merge decision. Use for contribution work; use mlzo-review alone for review-only requests.
---

# Prepare an MLZO change

Read the root AGENTS.md and [CONTRIBUTING.md](../../../CONTRIBUTING.md). Use the
existing Python and CLI contracts, not the future capabilities in plan.md.

## Establish the workspace

- Inspect `pwd`, `git status --short --untracked-files=all`,
  `git branch --show-current`, `git worktree list --porcelain`, and `git rev-parse HEAD`.
  Confirm the requested behavior, task-owned files, intended base, and branch.
- Reuse the current prepared worktree when it belongs to this change, including
  a resumed task with its own dirty files. A different branch name convention is
  not a reason to create another worktree. Do not switch or reset its branch.
- Otherwise create a new branch and worktree from the agreed base using the
  commands in CONTRIBUTING. If a candidate path or branch exists, inspect it.
  Reuse it only when its history and changes belong to this task; otherwise pick
  a fresh name. Never use `--force` to take over another checkout.
- Uncommitted work in another checkout is not in the new worktree. Do not copy,
  stash, reset, delete, or commit it. Inspect it read-only for compatibility when
  useful, and describe the separate source revision if you test it.

## Implement and prove the behavior

Choose the smallest useful reproduction and acceptance check. For a bug fix,
observe that check fail on the base for the reported reason before changing
production code. Add complexity only when implemented behavior needs it.

Trace the real execution path and extend an existing test when it covers the
failure mode. Avoid wrapper duplicates, assertions about prose, or mock-call
mirrors. Explain any test removal and the remaining check that catches the same
bug. Do not change unrelated production code to make this workflow look green.

Run the focused check, then the small CPU suite and package checks in
CONTRIBUTING. Isolate state and bound subprocess/network work; clean up owned
resources even on failure. Capture commands, exit status, relevant observed
output, Python/PyTorch/platform versions, and unsupported or untested cases.
Do not claim a skipped check passed or treat a local socket test as multi-host
training evidence. Do not provision cloud resources as a routine test step.

## Review and hand off

Commit only task-owned files, inspecting the staged diff first. Record the base
commit, merge base, head commit, and `HEAD^{tree}` for a clean candidate. Include
dirty/untracked scope as described in mlzo-review if it remains; HEAD alone
does not identify uncommitted work.

Use [mlzo-review](../mlzo-review/SKILL.md) for a separate reviewer session or a
human reviewer. One independent pass is sufficient to start; no provider,
parallel reviewer group, paid bot, or numeric score is required. If only the
implementer can check the diff, call it a self-review and leave independent
review pending. Do not invent a reviewer or silently start delegates when the
task disallows them.

Fix demonstrated bugs, explain rejected findings with evidence, and rerun the
affected checks. Request another review only for changed behavior or unresolved
findings. Stop circular review when it produces no new evidence and report the
remaining disagreement to the human.

Prepare the PR template with actual evidence and review status. Push or create
or update a PR only within the user's existing authorization. Without it, leave
a local commit and a ready-to-use description. Human merge approval is a
separate decision about the checked revision; implementation or PR permission
alone does not authorize merging or auto-merge.

Before handoff or an authorized merge, verify the current base/head/tree,
required CI, review findings, and approval again. A new commit, rebase, merge,
base update, or working-file edit makes previous evidence stale. Assess the
change, rerun relevant checks, and refresh the review record for the new
candidate. Never carry an old green check or approval onto an unchecked tree,
or bypass protection. Leave the worktree for inspection; remove it only after
the change is retained elsewhere and cleanup is requested or already authorized.
