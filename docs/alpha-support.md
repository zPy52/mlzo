# Alpha support boundaries

`0.1.0a1` implements a small subset of the original [plan](../plan.md). Unsupported operations must fail explicitly. The plan's distributed, recovery, provisioning, and inference-server designs are future work.

| Area | Available in this alpha | Limits |
| --- | --- | --- |
| Training | Local CPU/device execution or one remote worker per run, supervised steps, custom `Training`, explicit named updates | No DDP, FSDP, tensor/pipeline parallelism, or multi-worker run |
| Planning | Deferred factories, device/precision decisions, placement checks | No forward execution, activation/memory probing, or automatic partitioning |
| Data | Native local sources, importable factories, JSON/JSONL, restricted tensor files, explicit image directories, optional S3 objects | No distributed sharding or exact arbitrary-iterator cursor resume |
| Checkpoints | Full local snapshots, checksums, atomic manifests, retention, compatible state restore | No S3 storage, per-block/sharded state, automatic recovery, or generation fencing |
| Resume | Local restoration of model/optimizer/scheduler/RNG/algorithm state and counters | Partial epochs restart; no exact sampler/worker replay; one-shot sources and remote resume unsupported |
| Cluster | Authenticated coordinator, expiring invitations, persistent scoped worker identity, source snapshots, scalar events, reconnect/cancel, completed-run checkpoints/exports | Trusted single-owner operation, one job/device per worker, no managed provisioning, HA scheduling, or automatic worker-failure recovery |
| Inference | Immutable local artifacts, strict weight loading, local synchronous prediction | No remote inference server or autoscaling; weights-only exports need a factory |
| Recipes | Single-environment DQN with bounded replay and target updates | Discrete actions and vector observations only; environment resets on resume; no PPO, vector environments, or distributed rollouts |

Python 3.12+ and PyTorch `>=2.2.2,<3` are declared requirements. The Intel macOS environment uses the legacy 2.2.2 wheel with NumPy below 2. Other platforms do not receive that NumPy upper bound. This is not a claim that the full Python/PyTorch range works. Optional Pillow, boto3, and Gymnasium imports happen on use. The worker supervisor requires POSIX process groups and file locking.

Validation includes compile/import/package checks, CPU examples, regression tests against PyTorch, and a localhost worker workflow. From the repository root, install the test dependencies with `python -m pip install -e '.[test]'`, then run `python -m pytest -q`; [CONTRIBUTING.md](../CONTRIBUTING.md) covers the CI workflow and package checks, and the [readability follow-up](readability-follow-up.md) records the local evidence. These checks are not a comprehensive numerical or platform validation matrix. GPU execution, CUDA mixed precision, MPS, remote GPU placement, TLS deployment, and optional S3/environment integrations remain unvalidated unless a release's recorded checks state otherwise. Do not infer validation from an available option.

Remote projects, dependency build/install code, and artifact factories execute trusted Python. Restricted loading avoids arbitrary pickle object loading, but it does not sandbox that code. Checksums detect content changes and do not establish authorship. Keep admin, submitter, enrollment, and worker credentials separate; protect coordinator and worker state directories. These services do not provide a hostile-code sandbox or a multi-tenant security boundary.

Checkpoint/export references returned for remote jobs identify paths on the assigned worker. Keep that worker's state and service to retrieve or export results. A client disconnect does not implement automatic failover. A worker restart can reconcile retained local state, but lost workers and ambiguous work require owner intervention.

The package declares the repository's MIT license and includes it in both distributions. Before publishing, verify the package name and release version and review both distributions. [Publishing](publishing.md) lists the commands; no upload is part of this implementation.

The initial alpha checks passed on Python 3.12.1 and PyTorch 2.2.2 on Intel macOS: compilation, package imports, CLI help/version, both CPU examples, local checkpoint/export/prediction, and one localhost worker job with reconnection, checkpoint copy, and export. The wheel and source archive passed `twine check`; the installed wheel passed an import and CLI check. Module agents also ran small smoke checks for deferred construction, data transforms, and DQN. These checks do not validate a remote Linux GPU or the full dependency range.
