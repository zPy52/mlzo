---
title: Training configuration
description: All train and fit options, defaults, validation rules, and support limits.
---

## train options

`mlzo.train(model_or_training, data=None, **options)` returns a deferred `Run`. The first input is an `nn.Module`, a `Training` instance, or an importable zero-argument training factory. A model needs data and either `loss=` or `step=`. A `Training` instance/factory supplies its own step and rejects separate loss/step arguments.

| Option | Default | Contract |
| --- | --- | --- |
| `loss` | `None` | Built-in loss name or callable `(prediction, target)`; mutually exclusive with `step` |
| `step` | `None` | Callable `(model, batch)` returning a scalar loss, Step, or None |
| `optimizer` | `"adamw"` | Built-in name, optimizer spec/class/factory/instance, or named mapping |
| `lr` | `None` | Optional finite nonnegative override, consistent with explicit optimizer rates |
| `schedule` | `None` | Schedule spec, name, callable, native scheduler factory/instance, or named mapping |
| `precision` | `"auto"` | `auto`, `fp32`, `bf16`, `fp16`; device restrictions apply |
| `matmul` | `"auto"` | `auto`, `strict`, `fast`; fast permits TF32, auto/strict use strict settings |
| `grad_clip` | `None` | Optional finite nonnegative gradient-norm limit |
| `accumulate` | `1` | Positive integer algorithm iterations per automatic update window |
| `compile` | `False` | `False`, `True`, or `"auto"`; compiles model forward only |
| `memory` | `None` | Only `None` or advisory `"auto"`; no memory probing or OOM recovery |
| `checkpoint` | `None` | Validated full `CheckpointPolicy` |
| `device` | `None` | Optional execution device constraint; must agree with explicit `on=` |
| `seed` | `None` | Integer engine/model seed; source and custom transform randomness may be separate |

Built-in optimizer names use learning rate `3e-4` unless specified; AdamW uses weight decay `0.01`. Other optimizer defaults follow PyTorch. Boolean values are not accepted as numeric budgets, accumulation, seeds, or rates.

## fit options

```python
run.fit(epochs=None, steps=None, on=None, parallel="auto",
        code=None, resume=None, wait=True)
```

All arguments are keyword-only. `epochs` and `steps` are mutually exclusive positive integer total targets. With neither supplied, ordinary data-driven training uses its default epoch budget; step-driven algorithms require a finite step target. Prefer specifying the budget explicitly.

| Option | Meaning |
| --- | --- |
| `on` | Local device string/torch.device or Cluster handle |
| `parallel` | `auto` or `single` for supported single-device execution. `None` and `"none"` are also accepted |
| `code` | Project directory for remote submission; invalid locally |
| `resume` | Local compatible checkpoint reference/directory, or `"latest"` with a policy; invalid remotely |
| `wait` | Must be boolean. Local execution requires True; remote False returns the submitted handle |

A prepared local run cannot be submitted remotely or change devices. An existing remote run cannot be resubmitted or extended. Unsupported distributed strategies fail explicitly.

## Precision and numerical settings

Automatic precision selects BF16 only on capable CUDA and FP32 otherwise. CPU FP16 and MPS mixed precision are rejected. Explicit CPU BF16 depends on operator support. GPU, mixed precision, and compilation remain unvalidated for this alpha.

MLZO scopes and restores process-wide matmul settings around execution. Concurrent MLZO execution in one process is rejected because these settings are shared. `compile="auto"` can retain eager execution after a compilation setup failure, but a failed training invocation is never retried, since it could repeat side effects.

## Configuration locations and environment

| Setting | Purpose |
| --- | --- |
| `MLZO_HOME` | Default configuration root, otherwise `~/.mlzo` |
| `--home` | Command-specific root override |
| `--directory` | Explicit coordinator or worker persistent state directory |
| `MLZO_TOKEN` | Client/admin credential fallback where documented by the CLI |
| `MLZO_SUBMITTER_TOKEN` | Coordinator submitter credential fallback |
| `MLZO_INVITE_TOKEN` | Worker enrollment invitation fallback |

Use protected token files for routine CLI input. Do not put credentials in project code, command positional arguments, or committed environment files. See the [CLI reference](../cli/) for exact flags and the [Python API](../api/training/) for complete signatures.
