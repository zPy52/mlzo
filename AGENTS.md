# Working on MLZO

- For an implementation task, use [mlzo-change](.agents/skills/mlzo-change/SKILL.md).
  For a review, use [mlzo-review](.agents/skills/mlzo-review/SKILL.md) on its own.
  [CONTRIBUTING.md](CONTRIBUTING.md) has setup commands and the human merge gate.
- Use one branch and worktree per change. Reuse a prepared worktree that belongs
  to the task. Preserve unrelated staged, unstaged, untracked, and ignored files.
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
- Do not publish, provision resources, change GitHub settings, or merge unless
  the user's task authorizes the action. Keep credentials and runtime artifacts
  out of commits and reports.
