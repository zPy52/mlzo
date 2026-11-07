# Contributing to MLZO

Keep each change focused on one problem. Describe the expected behavior, prove
it with a small Python or CLI example, obtain a separate local review, and let
a human decide whether to merge. We use local review for the same purpose as
hosted review bots, to investigate likely bugs. We do not claim equal coverage
or assurance, and no paid review service is required.

MLZO currently trains on one local device or one enrolled remote worker per
run. See [alpha support](docs/alpha-support.md). A coordinator can manage workers
without implementing training across them. `plan.md` describes future work.
This guide supersedes the original no-test-suite instruction in the historical
[alpha implementation contract](docs/alpha-contract.md#validation-and-limits).

## One worktree and branch per change

First inspect `git status --short --untracked-files=all` and
`git worktree list --porcelain`. If T3 or another tool already prepared a
worktree and branch for this task, use them. Do not create a second worktree
just to match these example names. Resume task-owned dirty work in place.

Otherwise, after choosing the base and fresh names:

```sh
git fetch origin main
git worktree add -b fix/checkpoint-restore ../MLZO-checkpoint-restore origin/main
cd ../MLZO-checkpoint-restore
```

Inspect an existing candidate path or branch before reuse. Reuse it only for
the same change. Never reset it, force a checkout, or move another task's
uncommitted work. A worktree starts from a commit; it does not include dirty
changes in the original checkout. Give each worktree its own environment,
MLZO_HOME, coordinator/worker state, and output directories. Bind test servers
to loopback port 0 so the OS chooses an unused port.

## Local checks

Use Python 3.12 and an environment inside this worktree:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
# Linux: install CPU-only PyTorch before installing MLZO.
python -m pip install 'torch>=2.2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,dev]'
python -m pip check
python -m pytest -q
python -m build
python -m twine check dist/*
```

On macOS, omit the CPU-index command and let the editable install select the
platform wheel. Intel macOS uses PyTorch 2.2.2 and NumPy below 2. The full
declared Python/PyTorch/platform range is not a tested compatibility matrix.
Installation and builds need a package index; the tests use synthetic local
data and no model/dataset downloads, cloud keys, GPU, or optional extras.

Check the wheel, not just the editable source. In a disposable environment
with PyTorch installed, install `dist/*.whl` with `--no-deps`, change outside
the checkout, then run `python -I -m mlzo --help` and `mlzo --json --version`.
CI does this after replacing its editable install with the built wheel.

The default suite checks deferred construction;
loss, gradients and updates against raw PyTorch with uneven accumulation;
epoch-boundary resume with optimizer, schedule and RNG state plus corruption
rejection; failed-step gradient cleanup; enrollment and credential scope;
single-worker remote training and reconnect; terminal failure followed by
another job; and CLI configuration success/failure without credential leaks.

The transport tests run a real coordinator in the test process and a worker
and its training jobs in separate local processes. They use temporary state,
OS-assigned ports, finite job budgets, bounded waits, and teardown that stops
owned process groups even after failure. The worker prepares a local venv;
the fixture project declares no dependencies to install. Polling waits for
terminal records through the alpha's polling API, not a fixed sleep that
assumes work finished. Tests have a 90-second ceiling. Worker tests require
POSIX; CI runs all tests on Linux with Python 3.12 and CPU wheels.

Each test owns a unique directory under `.pytest_cache/d/mlzo-runtime` and
removes it on teardown. This also isolates overlapping test runs; cleanup
must not remove another run's coordinator state.

Local checkpoint comparisons share a filesystem. These checks do not certify
inter-machine layer/gradient exchange, distributed backpropagation, TLS, GPU
correctness, OOM recovery, IAM/VPC behavior, or production readiness.

For a bug fix, reproduce the failure before modifying production code. Add the
smallest test that catches it, preferably in an existing case. Extend the suite
when an implemented behavior needs protection. Remove redundant tests only
after naming the same failure mode another retained check catches. Skip tests
that repeat implementation, check prose, or merely count mock calls.

## Review, evidence, and the merge decision

Codex discovers the project-local [skills](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)
under `.agents/skills`. Ask it to use `$mlzo-change` for a change or
`$mlzo-review` for review alone. Other agents can read the linked SKILL.md files
directly. A separate reviewer session or a human can do the independent pass.
The implementer's self-review is useful but must be labeled as such. Record
independent review as pending if none occurred; do not invent an approval.

Give the reviewer the intent, base/head SHAs, merge base, and relevant checks.
For a clean candidate record `git rev-parse HEAD 'HEAD^{tree}'`. For local edits
include staged/unstaged diffs and in-scope untracked file hashes, as specified
in [mlzo-review](.agents/skills/mlzo-review/SKILL.md). Do not hide dirty inputs
behind a clean HEAD SHA or stage someone else's work to obtain a fingerprint.
Review should trace real callers and reproduce likely bugs. Findings should
identify affected lines, a trigger, a consequence, and supporting evidence.
Separate pre-existing bugs and unverified concerns from introduced defects.

Fix demonstrated bugs and rerun affected checks. Explain rejected findings
with evidence. A single separate review is enough to start; further rounds
need new changes or unresolved findings. Stop circular disagreement and give
the human the evidence instead of chasing a score.

Commit only this change, then fill the [PR template](.github/pull_request_template.md)
with exact commands, outcomes, environment versions, revision/tree, review
status, and unverified behavior. Terminal transcripts and numerical comparisons
are the evidence for this CLI/Python project. No screenshots or videos are
needed. Do not commit runtime state, credentials, logs, or PR-only artifacts.

Push and create/update a PR only when the task authorizes it. Otherwise leave
the local commit and prepared description. Before merging, check the current
base/head, required CI, unresolved findings, and human decision. New commits,
base updates, rebases, merges, or dirty edits invalidate previous evidence.
Refresh relevant checks and review for the resulting candidate. GitHub tests a
synthetic PR merge commit by default; distinguish that SHA from the branch head
when reporting CI. A green run or agent report does not authorize merging.

A human must decide to merge the checked revision. An agent can perform that
action only with explicit merge authorization for it. Never enable auto-merge
or bypass protection as part of this workflow. After the change is retained in
the merged history, remove a clean task-owned worktree with `git worktree remove`
and its merged branch with `git branch -d` when cleanup is authorized. Inspect
untracked/ignored artifacts first; do not force deletion if either command refuses.

## Enable the GitHub gate after the first run

The workflow runs on every PR and pushes to main, with no path filters. It has
read-only contents access, no secrets or OIDC permission, no stored checkout
credential, a job timeout, and stale-run cancellation. It uses `pull_request`,
never `pull_request_target` to execute submitted code. No branch rules are
installed by these files.

After the first successful run, a repository admin should open Settings >
Branches > Add branch protection rule for `main`. Require a pull request and
the check named `CPU tests and package`, selecting GitHub Actions as its source.
Require branches to be up to date and conversations to be resolved, and disable
force pushes/deletion. Apply the rule to administrators too. With another
eligible human maintainer, require one approval and dismiss stale approvals.
A sole owner cannot approve their own PR, so keep the human merge decision
manual until there is a second reviewer instead of configuring an unusable
self-approval gate. These rules require an admin and an eligible GitHub plan.
See [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Future opt-in two-EC2 check

No cloud workflow or resources are created here. When remote deployment needs
verification, add a manual `workflow_dispatch` job for a reviewed immutable
commit SHA. Use a protected environment with human approval. Use GitHub OIDC
to assume a restricted test-account role, not stored AWS access keys. Restrict
its trust to the actual repository/environment subject and audience, give
`id-token: write` only to that job, and keep untrusted PR code out of any job
with cloud authority. See [GitHub's AWS OIDC guide](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws).

Start with two small CPU instances, one coordinator and one worker, a tiny
synthetic MLP, and verified TLS between hosts. Bound the run to two instances,
one allowed instance type/region, a 15-minute job and 30-minute resource TTL.
Tag resources by run ID and expiry, cap cost, restrict network access, and
remove instances, volumes, and temporary network resources on success, failure,
or cancellation. Use an independent TTL cleanup mechanism for an interrupted
runner. Review IAM and cleanup before enabling it.

That test would validate real connectivity and the existing single-worker
contract. It would not be a two-worker training test. Add layer/gradient
exchange and distributed backprop tests only when those capabilities exist.

## Upstream ideas

This is a small MLZO adaptation of T3's
[behavioral evidence](https://github.com/pingdotgg/t3code/blob/main/CONTRIBUTING.md#verification)
and [focused testing](https://github.com/pingdotgg/t3code/blob/main/docs/internals/testing.md),
and Anthropic's
[investigation and human review](https://claude.dev/blog/solving-code-review-at-anthropic/).
Their infrastructure and reported results are not guarantees for this repository.
