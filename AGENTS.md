# Working on MLZO

- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for implementation and test commands.
  Use [mlzo-file-pr](.agents/skills/mlzo-file-pr/SKILL.md) to gather evidence and
  file a PR, or [mlzo-review](.agents/skills/mlzo-review/SKILL.md) when a review
  is requested. A separate agent review is optional.
- Every task thread must have its own branch and worktree. Reuse a prepared
  worktree that belongs to that task, including when resuming it. Preserve
  unrelated staged, unstaged, untracked, and ignored files.
- The implemented contract is local single-device training or one remote worker
  per run. Read [support boundaries](docs/alpha-support.md) before changing it.
  `plan.md` is a future design, not implemented behavior.
- Keep planning deferred, checkpoints restricted to tensors and primitives, and
  optional dependencies lazy. Unsupported operations must fail explicitly.
- Prefer a small behavioral regression test over mocks of internal calls. Remove
  a test only with evidence that another test covers its failure mode. Run
  `python -m pytest -q` and the package checks in CONTRIBUTING before handoff.
- Use CLI output, numerical comparisons, and logs as evidence. Local CPU and
  loopback tests do not validate multi-host backprop, GPUs, OOM recovery, TLS,
  IAM/VPC configuration, or production readiness.
- The owner reviews every PR and personally accepts, rejects, or merges it.
  Leave PRs open for that decision. Do not merge, enable auto-merge, or bypass
  branch protection as part of preparing a PR.
- Push or file a PR only when requested. Do not publish, provision resources,
  or change GitHub settings unless the task authorizes it. Keep credentials
  and runtime artifacts out of commits and reports.
