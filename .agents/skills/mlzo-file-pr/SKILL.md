---
name: mlzo-file-pr
description: Gather test evidence and file or update an MLZO pull request for the owner to review. Use when asked to prepare, file, open, or update a PR.
---

# File an MLZO PR

Read [AGENTS.md](../../../AGENTS.md) and
[CONTRIBUTING.md](../../../CONTRIBUTING.md) for workspace rules, test commands,
and contribution policy. Finish the requested change with evidence the owner
can review. The owner personally accepts, rejects, or merges every PR.

## Establish the candidate

Use this task thread's branch and worktree, reusing a prepared workspace for
the same task. Inspect status and ownership before modifying or committing
anything. Preserve unrelated staged, unstaged, untracked, and ignored files.

Identify the repository, intended base, branch head, and merge base. Check for
an existing PR for this branch before creating one; use its actual base rather
than assuming main. Fetch the relevant remote refs when available and disclose
if the live base cannot be verified. Inspect the complete diff and in-scope
local files against the base. Confirm they address the requested problem.

## Gather evidence

Run the focused reproduction or acceptance check, then the full CPU suite and
package checks in CONTRIBUTING, including installation of the built wheel
outside the checkout. For a bug fix, show the reproduction failing before the
fix and passing afterward. For documentation or skill changes, inspect the
instructions, links, and skill metadata instead of adding tests of wording.

Record exact commands, exit status, relevant output, Python/PyTorch/OS/device,
and untested behavior. Distinguish failures, skips, and blocked checks from
passes. Use terminal output, numerical comparisons, and relevant log excerpts;
screenshots and videos are unnecessary. CPU and loopback checks do not prove
GPU, multi-host, TLS, or AWS behavior. Keep runtime artifacts and credentials
out of commits and reports.

Self-review the diff and fix demonstrated bugs within scope. A separate agent
review is optional; use [mlzo-review](../mlzo-review/SKILL.md) when requested.
Do not start reviewers, review loops, or a paid review service just to file a PR.

## File or update the PR

Inspect the staged diff and commit only task-owned files. Record the base,
merge base, head SHA, and `HEAD^{tree}` for the committed candidate. If relevant
files change after checking, inspect the delta and refresh affected evidence.

Use the [PR template](../../../.github/pull_request_template.md). Lead with the
concrete problem and resulting behavior; keep the title and explanation short.
Fill in the actual evidence, self-review or requested-review status, remaining
risks, and owner review pending. Check the entire description for secrets and
private details before posting.

An explicit request to file or update a PR authorizes pushing this task's
branch and creating or editing that PR. If only preparation was requested,
leave a local commit and prepared description. Update an existing PR rather
than filing a duplicate. In T3 Code, link a created or updated PR with
`link_pull_request` immediately after creating it or pushing to it.

## Hand off

Verify the PR URL, head SHA, base, and posted description. Report the PR link,
checks and their checked revision, remaining gaps, and owner review pending.
Hand it to the owner without starting a watcher or automated review process.
Leave it open and unmerged; never enable auto-merge or bypass branch protection.
