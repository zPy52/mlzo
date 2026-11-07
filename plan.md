MLZO should let a developer define an ordinary PyTorch model, describe how it learns, and train or deploy it on local or remote GPUs while the framework handles execution, data movement, backpropagation, and infrastructure.

# MLZO product specification and implementation plan

Status: proposed design. No framework implementation exists in this repository yet.

This document consolidates the supplied `pasted-text.txt` specification and the clarification that a MacBook must be able to coordinate a cluster containing even one remote GPU, without SSH for everyday training. Part I defines what to build and what MLZO handles automatically. Part II describes how to implement it, in dependency order.

All MLZO examples describe the intended API. They are not claims that the package already provides these capabilities. Additions that resolve gaps in the original discussion are identified as proposed refinements. External integration references were checked on 2026-11-07; supported dependency versions must be pinned and tested during implementation.

Contents:

- [Part I: What to implement](#part-i-what-to-implement)
- [Part II: How to implement it](#part-ii-how-to-implement-it)
- [Implementation milestones](#24-implementation-milestones)
- [Acceptance and release gates](#25-acceptance-and-release-gates)
- [Decisions requiring validation](#26-decisions-requiring-validation)

---

# Part I: What to implement

## 1. Product goal and fixed design decisions

MLZO is a Python framework and distributed runtime for creating, training, maintaining, and deploying machine-learning systems. The comparison with Mojo expresses the ambition to make a familiar ecosystem much easier and more capable. The initial implementation remains Python and builds on PyTorch.

The developer should spend time on model architecture, objectives, data, and learning rules. Writing an ordinary training job should not require coordinating CUDA devices, process groups, gradient synchronization, pipeline communication, mixed precision, checkpoint shards, or remote shells.

The following decisions carry through the entire plan:

1. Models remain ordinary `torch.nn.Module` objects. There is no replacement model superclass.
2. Declarative arguments, ordinary functions, and `mlzo.Training` classes share one execution engine.
3. `step()` is the only normally required training-class override. Initialization belongs in `__init__()`.
4. Execution options belong to `mlzo.train()` and `Run`, rather than mandatory lifecycle callbacks.
5. A scalar loss requests automatic optimization. Explicit `self.update()` calls express unusual update ordering.
6. `mlzo.Lazy(factory, ...)` postpones module construction. Users do not write a string-based model language.
7. `gpu=` and `mlzo.place()` describe placement without allocating or moving weights.
8. The coordinator can be a CPU-only laptop with insufficient RAM to hold the model.
9. Workers own the heavy computation and communicate tensors directly with other workers.
10. Python and CLI interfaces use the same cluster and run-management services.
11. Plans explain execution decisions, and unsupported combinations fail before expensive execution when detectable.
12. Mathematical correctness takes precedence over an optimization that changes the learning algorithm.

The first backend is PyTorch on Python 3.12+. Initial production GPU workers run Linux with NVIDIA CUDA. macOS is a supported development and coordinator platform. CPU execution supports development and tests. Other accelerators become supported only after backend-specific validation.

Do not initially build a tensor engine, autograd engine, CUDA compiler, communication library, general-purpose cluster scheduler, or new programming language. JAX, TPU execution, and a possible language extension remain later investigations.

## 2. The central workflow: write locally, compute remotely

### 2.1 A cluster can contain one GPU

A single rented GPU is a complete supported cluster. It must not require multi-GPU configuration or a special remote-training API.

The intended workflow is:

1. The developer writes model, loss, and data code on their MacBook.
2. A remote machine runs an enrolled MLZO worker service. A managed provider can install and start it automatically.
3. The developer runs a local Python script and selects the cluster with `run.fit(on=cluster)`.
4. MLZO snapshots the project, prepares a compatible worker environment, and stages any declared local data.
5. The worker constructs the model, loads data and checkpoints, and executes the entire training loop.
6. The MacBook receives status, logs, metrics, and artifact references.
7. The developer evaluates, resumes, exports, or deploys through the same local Python interface.

No SSH session, manual code copying, remote `pip install`, remote `torchrun`, or manual restart command should be required for each run. An existing unmanaged machine still needs initial worker installation and enrollment. This bootstrap can happen through a prepared image, cloud startup script, administrator, or one-time installation; subsequent training uses MLZO.

### 2.2 One remote GPU example

The project contains an importable model factory. Calling `mlzo.Lazy` on the laptop does not call this factory.

```python
# project/models.py
from torch import nn


def make_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
```

```python
# train.py, executed on the MacBook
import mlzo
from project.models import make_classifier


def main() -> None:
    cluster = mlzo.cluster("my-gpu")
    model = mlzo.Lazy(make_classifier)
    data = mlzo.data("s3://my-datasets/digits/train/").batch(64)

    run = mlzo.train(
        model,
        data,
        loss="cross_entropy",
        checkpoint=mlzo.checkpoint(
            "s3://my-results/digits/",
            every=1000,
        ),
    )

    run.fit(epochs=10, on=cluster, code="./")
    print(run.status())


if __name__ == "__main__":
    main()
```

The dataset source in this example must have a supported format containing input and target records. MLZO reads its manifest and schema; it must not guess labels from arbitrary files. Worker credentials grant access to the indicated storage.

`code="./"` makes the project boundary explicit. The eventual default discovers the project from `pyproject.toml` or the script location and packages it automatically. It fails with a useful message if that boundary is ambiguous. The `if __name__ == "__main__"` guard prevents importing the script on workers from submitting another job.

### 2.3 Multiple GPUs and direct tensor movement

The same workflow extends to several GPUs on one machine or across machines. If GPU 1 produces an activation needed by GPU 2, MLZO sends that activation through the worker data network. During backpropagation, the corresponding activation gradient travels back through that network.

The coordinator does not receive the activation, copy it into laptop RAM, or forward it to another GPU. It also does not perform the forward pass or orchestrate each tensor operation through a Python RPC.

"Direct" means the coordinator is absent from the tensor path. Depending on the hardware, transport can use GPU peer access, NVLink, PCIe, RDMA, or worker-host buffers and sockets. MLZO should use the supported efficient path and report the transport it selected. It cannot promise physical GPU-to-GPU DMA on every network.

### 2.4 What lives where

| Information or work | Development machine / coordinator | Worker machines |
|---|---|---|
| Project editing and submission | Owns source before submission | Receive an immutable project artifact |
| Model description and placement | Stores factories, hierarchy, and metadata | Resolve assigned descriptions |
| Parameters and model buffers | Not required for a fully lazy model | Allocate, initialize, load, and update |
| Optimizer tensors and gradient buffers | Not required | Own and checkpoint |
| Dataset reads and preprocessing | Describe sources; upload local inputs once if needed | Read, transform, shard, and prefetch |
| Forward computation and loss | Not on the training path | Execute |
| Backpropagation and optimizer updates | Not on the training path | Execute |
| Activations and activation gradients | Never a required relay | Exchange between participating workers |
| Checkpoint contents | Coordinate metadata; optional explicit downloads | Write directly to checkpoint storage |
| Plans, logs, metrics, run status | Inspect and display | Produce and report |
| Inference after deployment | Manage the deployment | Load artifacts and serve requests |

This guarantee excludes weights or data that the developer explicitly creates on the laptop before giving them to MLZO. An eager `nn.Linear(...)` already allocates parameters. The framework cannot undo that allocation; `Lazy` is the route to a coordinator without model weights.

### 2.5 The laptop is not a permanent compute dependency

Once a job is accepted, workers execute its bounded training program without asking the laptop for permission on each step. A temporary control connection loss must not interrupt healthy worker-to-worker computation. Workers buffer events and continue checkpointing to a reachable destination.

If the laptop also hosts the only coordinator, its absence temporarily prevents new scheduling, membership changes, and replacement-worker decisions. The accepted fixed worker group may continue to its declared budget. Recovery or reassignment waits for a functioning coordinator. A persistent coordinator on a server or worker supports longer detached operation.

Proposed refinements are `run.fit(..., wait=False)` for submitting without following logs, and `mlzo.run(run_id, on=cluster)` for reconnecting. Blocking `fit()` remains the default. These conveniences do not imply automatic coordinator high availability.

## 3. What MLZO must handle by default

The default contract is about removing repetitive work. It does not mean enabling every available optimization or choosing an unspecified loss function.

| Responsibility | Automatic behavior | What the developer supplies or may override |
|---|---|---|
| Run construction | Validate configuration and produce a `Run` without starting training | Model or training algorithm, data, and objective |
| Model construction | Materialize lazy modules on their assigned workers | A callable factory and its arguments |
| Device selection | Use a suitable available local device or the selected remote cluster | An explicit local device or cluster |
| Input movement | Move batch tensors to the consuming worker/device | Custom handling for unusual batch objects |
| Forward execution | Call the model in the correct execution context | Custom forward/loss logic through `step()` |
| Backpropagation | Calculate parameter and input gradients for a returned loss | Selective `no_grad`, detach, or explicit update rules |
| Gradient clearing | Clear gradients at the correct update boundary | No ordinary `zero_grad()` call |
| Optimizer updates | Bind actual worker parameters and apply the configured optimizer | Optimizer, parameter groups, and hyperparameters |
| Mixed precision | Resolve a supported conservative policy and manage autocast/scaling | `precision="fp32"`, `"bf16"`, or `"fp16"` |
| Accumulation | Normalize and accumulate contributions; handle a short final window | `accumulate=N` and custom weighting if needed |
| Clipping | Unscale first and compute the appropriate gradient norm | `grad_clip`, disabled unless requested |
| LR scheduling | Advance at the documented successful-update boundary | Optional schedule; none by default |
| Distributed initialization | Assign ranks, devices, groups, and rendezvous configuration | Select a cluster and optional strategy |
| Data parallel synchronization | Combine gradients across the correct replicas | No manual collective calls |
| Pipeline communication | Send activations forward and gradients backward between workers | Supported model structure or explicit stage definitions |
| Pipeline scheduling | Split batches, respect dependencies, and coordinate stage work | Optional stage layout and microbatch count |
| Code distribution | Package the project and dependencies; prepare/cache worker environments | Importable code and declared dependencies |
| Remote data access | Load remote sources on workers; stage declared local data once | Dataset source and credentials/access policy |
| Checkpoint state | Save model, optimizer, scheduler, scaler, counters, and registered state together | Destination, frequency, and retention policy |
| Monitoring | Record run identity, loss, update counts, status, and failures | Optional metric functions or external integrations |
| Failure handling | Detect failures, fence stale work, and resume from a valid snapshot when configured | Retry/recovery policy and a durable checkpoint destination |
| Run cleanup | Close run processes and temporary resources according to ownership | Whether owned cloud resources remain available |
| Deployment loading | Load the model artifact and prepare an inference worker | Artifact and deployment target |

Proposed baseline defaults:

- `optimizer="adamw"`, `lr=3e-4`, `accumulate=1`, and no LR schedule or gradient clipping. The built-in AdamW specification uses `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, and `amsgrad=False`. Kernel choices remain backend-controlled and recorded. These are convenient defaults, not universal hyperparameters.
- `precision="auto"` resolves to BF16 on validated capable CUDA configurations and FP32 elsewhere. Initial auto mode does not select FP16 without an explicit choice.
- `compile=False` initially. Compilation and memory-saving policies remain opt-in until their supported combinations have been measured.
- `matmul="auto"` initially uses the strict FP32 policy. `matmul="fast"` opts into the documented numerical tradeoff independently of AMP.
- `parallel="auto"` selects single-device execution for one available GPU. It chooses only strategies validated for the model, algorithm, and topology.
- Basic metrics and error reporting are always active. Checkpoint writes require a destination/policy; persistent run metadata alone does not guarantee recoverable model state.
- A finite dataset with no explicit fit limit gets one epoch. An unbounded source requires a finite `steps` or epoch boundary definition.

Automatically choosing a strategy must not silently change the effective global batch, objective, optimizer, sample coverage, or update ordering. MLZO can propose a different batch size after an OOM; it must not quietly apply it to a stateful run.

## 4. Training API and customization

### 4.1 Declarative training

```python
run = mlzo.train(
    model,
    train_loader,
    loss="cross_entropy",
    optimizer="adamw",
    lr=3e-4,
)
run.fit(epochs=10)
```

The standard supervised batch is `(inputs, targets)`. MLZO calls `model(inputs)` and then the configured loss. Other input structures use a documented input adapter or a custom step; MLZO does not infer the meaning of arbitrary dictionary keys.

Built-in losses retain PyTorch shape and mathematical conventions. In particular, cross entropy does not silently infer a class dimension or shift language-model targets. A sequence-model example must explicitly flatten logits/targets or use a named task recipe that documents the shift and masking.

A callable loss is an ordinary differentiable function of prediction and target. It returns a scalar tensor. The caller specifies any class weighting, reduction, or masking whose meaning the framework cannot determine.

### 4.2 Functional training

```python
import torch.nn.functional as F


def step(model, batch):
    inputs, targets = batch
    predictions = model(inputs)
    return F.cross_entropy(predictions, targets)


run = mlzo.train(model, train_loader, step=step)
run.fit(epochs=10)
```

The function controls forward computation and loss construction. MLZO still owns backward and optimizer mechanics. `loss=` and `step=` are mutually exclusive.

### 4.3 Class-based algorithms

```python
import torch
import torch.nn.functional as F


class RecursiveTraining(mlzo.Training):
    def __init__(self, model, cycles=8):
        super().__init__(model)
        self.cycles = cycles
        self.state = None

    def step(self, batch):
        inputs, targets = batch
        carry = None

        with torch.no_grad():
            for _ in range(self.cycles - 1):
                carry, _ = self.model(inputs, carry)

        carry, prediction = self.model(inputs, carry)
        return F.cross_entropy(prediction, targets)
```

This illustrates truncated recursive differentiation. It is not an exact HRM or TRM implementation. MLZO must eventually represent their actual update rules, including state and halting, through ordinary algorithm code.

No mandatory `initial_state()`, `configure_optimizers()`, or callback stack is required. Attributes hold configuration, `self.state` holds recognized checkpointable algorithm state, and ordinary methods organize the algorithm.

### 4.4 Step results and update ownership

A step can return a scalar loss, `mlzo.Step`, or `None`.

```python
result = mlzo.Step(
    loss=loss,
    metrics={"accuracy": accuracy.detach()},
)
```

The contract must be unambiguous:

- A loss return requests one automatic update, subject to accumulation.
- `Step(loss=None, metrics=...)` reports metrics without an automatic update.
- `None` means no automatic update. It is valid for an explicit-update algorithm or an iteration that only collects experience.
- Any `self.update(...)` call followed by a returned optimizable loss is an error. It cannot accidentally cause a second update.
- `Step.stop=True` requests run termination at the next consistent boundary. `False` and `None` continue.
- A tensor-valued stop mask is reserved for an explicit adaptive-batch policy with sample identities. Ordinary training rejects it; it does not silently interpret per-example halting as whole-run stopping.

Proposed refinement: `Step.weight` expresses how many loss units a scalar mean represents. It supports unequal batches, masked tokens, and custom reductions. The exact normalization contract appears in section 16.

### 4.5 Explicit updates and multiple optimizers

```python
class GANTraining(mlzo.Training):
    def step(self, batch):
        d_loss = self.discriminator_loss(batch)
        self.update("discriminator", d_loss)

        g_loss = self.generator_loss(batch)
        self.update("generator", g_loss)

        return mlzo.Step(metrics={
            "d_loss": d_loss.detach(),
            "g_loss": g_loss.detach(),
        })


run = mlzo.train(
    GANTraining(model),
    data,
    optimizer={
        "discriminator": mlzo.opt("adamw", params="discriminator", lr=2e-4),
        "generator": mlzo.opt("adamw", params="generator", lr=2e-4),
    },
)
```

`self.update()` completes its optimizer operation before returning. Therefore the next forward computation can observe the updated parameters. MLZO must not defer these operations to the end of `step()` because that would change alternating algorithms.

The algorithm controls detach/freeze decisions and computes fresh forward values after parameter changes. The runtime handles zeroing, backward, AMP, clipping, synchronization, and per-optimizer counters. Initially, overlapping parameter ownership between optimizers and explicit updates combined with `accumulate>1` are rejected. Their future semantics require a separate tested policy.

### 4.6 Evaluation and persistent state

`run.evaluate(data)` performs inference without optimizer updates and restores the previous train/eval mode afterward. Declarative training can reuse its loss. An arbitrary training `step()` may mutate state or perform updates, so evaluation must use an optional `evaluate` function or `Training.evaluate()` override. This is optional for training and required only for custom evaluation behavior.

Checkpointed state can contain tensors, primitive values, and supported containers. Replay buffers, environment state, external services, and other complex objects require an explicit adapter. Attributes are not blindly pickled.

Persistent reasoning state is keyed by sample or stream identity. The runtime must never attach one shuffled batch's hidden state to another batch. State carried across steps must have an explicit detach policy so it does not accidentally retain an unbounded autograd graph.

## 5. Optimization, transformations, and memory

The configuration should remain concise:

```python
run = mlzo.train(
    training,
    data,
    optimizer="adamw",
    lr=3e-4,
    schedule=mlzo.cosine(warmup=1000),
    precision="bf16",
    grad_clip=1.0,
    accumulate=4,
    compile="auto",
    memory="auto",
)
```

Support built-in optimizer names, PyTorch optimizer instances, factories, classes with validated arguments, parameter groups, and named optimizers. Existing instances are accepted only when they reference the actual parameters used for execution. Lazy, remote, or parameter-replacing strategies normally require a factory.

Schedulers include constant, linear, cosine, exponential, callable multipliers, and native PyTorch scheduler factories. `lr` is the base rate, and the peak rate for built-in warmup schedules. Each optimizer has its own successful-update counter. Overflow skips do not advance its update-based scheduler. Epoch-based and metric-based schedules have explicit policies.

Precision choices are FP32, BF16, FP16, and auto. AMP does not imply casting every model parameter to a low-precision dtype. Compilation is optional and reports graph breaks/fallbacks. `memory="auto"` may select tested activation-checkpointing or attention implementations, with the actual decisions listed in the plan. It cannot change a model's mathematical operations or external side effects without a documented contract.

Expose thin function-transform wrappers:

```python
df = mlzo.grad(f)
batched_df = mlzo.vmap(df)
compiled_df = mlzo.jit(df)
jacobian = mlzo.jacrev(f)
forward_jacobian = mlzo.jacfwd(f)
hessian = mlzo.hessian(f)
value, tangent = mlzo.jvp(f, primals, tangents)
value, pullback = mlzo.vjp(f, *primals)
```

These use PyTorch semantics. `grad` requires a scalar objective; Jacobian transforms support vector outputs. Transforming modules uses functional parameters/buffers where necessary. Side effects, randomness, and unsupported compositions produce diagnostics. This layer does not promise that arbitrary Python becomes a pure, compilable function.

## 6. Data and optional named axes

### 6.1 Data handling

Keep native `Dataset`, `IterableDataset`, and `DataLoader` support for local runs. Offer a composable pipeline for common work:

```python
data = (
    mlzo.data("images/")
    .images(size=(224, 224))
    .normalize("imagenet")
    .shuffle(seed=42)
    .batch(64)
)
```

Image-directory conventions must explicitly define label discovery. Generic image decoding alone does not create supervised targets.

Support `map`, collation, padding, caching, prefetching, CPU workers, streaming, deterministic random transforms, and optional GPU preprocessing. Text support begins with user-supplied tokenizers and documented padding/masking; MLZO must not select an arbitrary vocabulary or tokenizer silently.

Remote data pipelines must be reconstructible on workers. `mlzo.data(make_dataset)` is the proposed factory form for an importable dataset builder. Known native datasets/loaders can have adapters; arbitrary open Python iterators cannot automatically move to a remote process.

For a local path, MLZO creates a versioned data artifact and uploads it once before remote execution. For an S3 or worker-accessible URI, workers read directly. `code="./"` does not indiscriminately bundle datasets, credentials, checkpoints, or the developer's virtual environment.

Data parallelism partitions records across replicas. Pipeline stages in the same replica cooperate on the same logical examples. Evaluation must avoid counting duplicated padding samples. Checkpointable sources restore the next unconsumed record, not merely the loader's prefetched position.

Proposed batch contract: `.batch(N)` denotes the global logical batch before data-parallel partitioning. With `R` replicas, MLZO divides that batch across the replicas and preserves the total. Pipeline microbatches subdivide each replica's portion. For full unmasked batches, `accumulate=A` therefore gives an effective batch of `N * A`, independent of GPU count. A native loader keeps its documented batch meaning; the planner may rebuild it through a supported adapter to preserve that global size, but cannot silently multiply its batch by the replica count. Initially reject unsupported uneven partitions or use a different valid auto strategy.

### 6.2 Named dimensions

Named axes are an optional convenience:

```python
x = mlzo.tensor(values, axes=("batch", "heads", "time", "dim"))
x = x.align("batch", "time", "heads", "dim")
x = x.flatten(("heads", "dim"), into="features")
x = x.mean("time")
```

Validate unique axis names, dimension counts, reorderings, reductions, and broadcast compatibility. Preserve autograd and provide an explicit underlying-tensor accessor, proposed as `.raw`. Begin with a wrapper for supported operations; do not promise transparent propagation through every PyTorch operator.

Axis metadata may later help placement or sharding. Names alone cannot prove that an operation is safe to partition.

## 7. Lazy models and placement

### 7.1 Construction must truly be deferred

```python
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = mlzo.Lazy(nn.Embedding, 50000, 4096, gpu=0)
        self.blocks = nn.ModuleList([
            mlzo.Lazy(TransformerBlock, hidden=4096, heads=32, gpu=i)
            for i in range(10)
        ])
        self.head = mlzo.Lazy(nn.Linear, 4096, 50000, gpu=9)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
```

`TransformerBlock` is an importable project class. Constructing this outer model creates a module hierarchy and recipes. It must not construct any wrapped layer or allocate its parameters.

`Lazy` inherits `nn.Module`, supports `ModuleList` and `Sequential`, retains its factory/arguments, has a useful unresolved representation, and supports explicit local `materialize(device=...)`. Repeated materialization returns the same local module. Materializing remotely does not silently populate the coordinator's wrapper with remote tensors.

Before materialization, `model.parameters()` cannot expose nonexistent parameters. MLZO binds optimizers after construction and strategy transformations. Custom initialization is preserved, and every tensor must be initialized or restored before execution.

### 7.2 Placement is metadata

```python
mlzo.place(model.encoder, gpu=0)
mlzo.place(model.decoder, gpu=1)
mlzo.place(model.blocks, gpus=range(10))
```

Exactly one of `gpu` or `gpus` is allowed. Group placement maps the ordered direct children one-to-one and validates cardinality. GPU IDs refer to logical devices in the selected cluster, not the laptop's CUDA indices.

A later explicit placement on the same module overrides its earlier annotation. Descendants inherit a parent assignment unless they have a more specific explicit assignment. Parent annotations do not erase child annotations. A module that cannot be split across the resulting devices produces a planning error.

Stable block names such as `blocks.4` identify model state independently of GPU numbering. Shared modules and tied parameters retain shared identity; MLZO must not silently duplicate them during partitioning.

## 8. Distributed execution and inspectable plans

Support these execution strategies incrementally:

- Single-device local or remote execution.
- Data parallelism, with replicated models and disjoint input shards.
- Sharded training, with parameters, gradients, and optimizer state distributed through supported backend facilities.
- Tensor parallelism, for registered compatible operations.
- Pipeline parallelism, with model stages on different devices.
- Validated combinations of these strategies in later milestones.

`parallel="auto"` selects from the supported subset and explains the decision. It must not quietly discard explicit placement or turn a custom iterative algorithm into a different static schedule.

```python
strategy = mlzo.pipeline(
    stages=10,
    split="blocks",
    microbatches=32,
    schedule="1f1b",
)

plan = run.plan(on=cluster, parallel=strategy)
print(plan.explain())
run.fit(epochs=10, on=cluster, parallel=strategy)
```

The same configuration is used for planning and execution. `fit()` revalidates the plan if hardware, code, shapes, or configuration changed.

A plan reports device inventory, stage boundaries, placements, parameter ownership, data shards, precision, compilation, optimizer groups, schedules, accumulation, pipeline microbatches, communication paths, checkpoint destination, and known limitations. Memory figures are labeled as measured, estimated, or unknown. A plan is validated, partially validated, or awaiting specific runtime checks.

Supported sequential models should preserve their original `forward()`. Capturable branches and skip connections require a graph that records all cross-stage dependencies. Unsupported Python control flow uses explicit stage definitions or another strategy. Placement alone does not make an arbitrary Python method remotely executable.

Pipeline gradients travel upstream between stages; stage-owned parameters do not require an all-reduce across unrelated stages. Replicated parameters do require synchronization within their replica groups. Tied parameters across stages need a defined ownership/synchronization policy and are rejected until supported.

## 9. Cluster management, enrollment, and cloud resources

```python
cluster = mlzo.cluster("research-cluster")
invitation = cluster.invite(max_joins=10, expires_in=900)
```

A worker can join through Python:

```python
import os
import mlzo

worker = mlzo.worker.join(os.environ["MLZO_INVITE_TOKEN"])
try:
    worker.serve()
finally:
    worker.leave(graceful=True)
```

The invitation is a short-lived enrollment credential. Enrollment produces a scoped persistent worker identity. The cluster separately authorizes who can submit executable project code.

The worker service reports device inventory, memory, software versions, and health. It prepares environments, launches job processes, loads assigned modules, participates in training, checkpoints its state, and supports graceful departure. `serve()` is blocking unless an explicit service/background mode is selected.

Required management operations are `status`, `workers`, `invite`, `drain`, `remove`, `scale`, and `shutdown`. Draining stops new assignments and reaches a safe boundary for existing work. Removing revokes membership. Scaling changes provisioned capacity; it does not automatically resize an active pipeline mid-step. Shutdown releases owned resources and stops services without powering off externally supplied machines.

Cloud provisioning should eventually be as concise as:

```python
cluster = mlzo.cluster.aws(
    gpu="H100",
    count=1,
    region="us-east-1",
    max_hourly_usd=budget,
)
```

Proposed semantics: `count` is the requested GPU count; `cluster.scale(count)` uses the same unit. The provisioning plan shows the number and type of machines required. This resolves the original ambiguity between GPU count and worker-machine count. A request that can only be fulfilled by allocating extra GPUs must show that allocation and its cost before provisioning.

Provider adapters handle instances, supported driver/runtime images, private networking, enrollment, storage identity, quotas, spot interruption, and cleanup. Spending limits and a reviewable provisioning plan precede resource creation. GCP, Azure, Kubernetes, Slurm, and TPU adapters follow later.

## 10. Checkpointing, recovery, and run lifecycle

```python
policy = mlzo.checkpoint(
    "s3://research-bucket/run-01/",
    every=1000,
    mode="per_block",
    keep_last=3,
)
```

Checkpointing must restore the learning process, including parameters, buffers, optimizer/scheduler/scaler state, progress counters, RNG state, data position, supported custom state, code identity, and topology metadata.

`mode="full"` stores a complete logical snapshot, potentially in several files. `mode="per_block"` organizes that snapshot by stable block identities. Both require a consistent global snapshot; a folder of unrelated block weights is insufficient for resuming training.

Workers upload their state directly to local/shared storage or S3. A final manifest identifies the committed snapshot. Failed partial uploads are never advertised as complete. Retention respects pinned checkpoints and keeps a known recoverable snapshot until its replacement is committed.

```python
reference = run.checkpoint()
run.fit(steps=10000, on=cluster, resume="latest")
run.export(block="blocks.4", to="s3://research-bucket/exports/block-4/")
```

An exported block is a weights artifact, not a resumable run. Resuming with a changed topology requires a supported resharding path; otherwise MLZO explains the incompatibility.

`Run` owns validation, planning, submission, execution observation, checkpoints, evaluation, export, and cleanup. A remote `Run` is a handle to worker execution, not an in-memory copy of the trained model. Proposed states are `created`, `planning`, `queued`, `preparing`, `running`, `checkpointing`, `recovering`, `completed`, `failed`, and `cancelled`.

Monitoring includes progress, losses, learning rates, completed and skipped updates, throughput, memory, worker health, stage utilization, the last checkpoint, and actionable failures. Basic monitoring requires no external dashboard.

## 11. Advanced learning algorithms

The same training interface must handle supervised and self-supervised training, diffusion objectives, GANs, physics-informed losses, recursive reasoning, and reinforcement learning. Compatibility with an algorithm does not imply compatibility with every distributed strategy.

HRM/TRM validation must cover repeated forwards, selective gradients, deep supervision, persistent sample state, halting, and auxiliary objectives. Keep a simple recursive fixture for engine tests and separately validate actual published implementations with their source revisions recorded.

RL recipes produce `Training` objects. They own environment interaction, replay or rollout storage, targets/advantages, and algorithm-specific update ordering; the generic runtime owns supported execution mechanics.

```python
agent = mlzo.dqn(
    q_network,
    env="CartPole-v1",
    replay=50_000,
    gamma=0.99,
    exploration="epsilon_greedy",
    target_every=1000,
    batch=64,
)
run = mlzo.train(agent)
run.fit(steps=100_000)
```

Recipes report separate environment-step, algorithm-iteration, and optimizer-update counters. They must distinguish termination from time-limit truncation when computing bootstrap targets. The intended behavior follows the distinction documented by [Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

DQN, actor-critic, PPO, policy gradients, and off-policy variants are staged recipes, not mandatory dependencies of basic training. Distributed rollout workers and learner workers are separate roles. Environments and replay buffers need explicit resume guarantees.

## 12. Deployment is part of the product

Training should produce an artifact that can be evaluated and deployed without reconstructing an ad hoc inference project. The original discussion leaves this area less specified; the following is a proposed extension.

An inference artifact contains model weights, a construction recipe or supported exported graph, input/output schema, required preprocessing/postprocessing, code/dependency identities, and the numerical policy used for inference. It excludes optimizer and training-only state unless explicitly requested.

```python
artifact = run.export(to="s3://research-bucket/models/classifier-v1/")
deployment = mlzo.deploy(artifact, on=cluster)
print(deployment.status())
```

`mlzo.deploy()` is a later proposed API. Its first implementation loads the artifact on one worker GPU, validates representative inputs, warms the model, and exposes an authenticated inference endpoint. Loading, batching within limits, device placement, health checks, logs, and shutdown are framework responsibilities. The coordinator remains a management client; inference traffic goes to the serving endpoint.

Requests have declared schema and size limits. Deployment versions are immutable. A replacement becomes active only after readiness checks and can roll back to the previous artifact. Remote one-GPU inference comes before distributed inference, autoscaling, quantization, or serving arbitrary multi-stage models.

## 13. Python and CLI parity

| Python operation | CLI operation |
|---|---|
| `mlzo.cluster("x")` | `mlzo cluster create x` |
| `cluster.status()` / `cluster.workers()` | `mlzo cluster status x` / `mlzo cluster workers x` |
| `cluster.invite()` | `mlzo cluster invite x` |
| `mlzo.worker.join(token)` | `mlzo worker join TOKEN` |
| `worker.status()` / `worker.leave()` | `mlzo worker status` / `mlzo worker leave --graceful` |
| `cluster.drain(id)` / `cluster.remove(id)` | `mlzo cluster drain x ID` / `mlzo cluster remove x ID` |
| `cluster.scale(n)` / `cluster.shutdown()` | `mlzo cluster scale x N` / `mlzo cluster shutdown x` |
| `run.status()` | `mlzo run status ID` |
| `run.checkpoint()` | `mlzo run checkpoint ID` |
| `run.fit(resume="latest", ...)` | `mlzo run resume ID --checkpoint latest` |
| Checkpoint catalog service | `mlzo checkpoint list LOCATION` |
| `deployment.status()` | `mlzo deployment status ID` |

Provide `--help`, structured errors, progress output, and `--json` for automation. Support protected token files or environment variables so enrollment does not require exposing secrets in shell history. Every command calls the same service used by Python.

---

# Part II: How to implement it

## 14. Internal architecture and dependencies

Separate four responsibilities internally:

1. The public API describes a learning algorithm and its configuration.
2. The planner resolves that description against available execution capabilities.
3. A worker execution engine runs the accepted plan and owns mutable training state.
4. Cluster services manage job identity, workers, artifacts, and recovery.

The local engine and remote engine use the same training implementation. Remote execution adds packaging, worker preparation, and observation around that engine. Do not implement a second training loop inside the coordinator or CLI.

Start with this package structure, creating modules only as their milestone needs them:

```text
pyproject.toml
src/mlzo/
  __init__.py
  api.py
  training/
    algorithm.py          # Training, Step, declarative/function adapters
    run.py                # Local execution or remote run handle
    engine.py             # Iteration and update state machines
    optimizers.py         # Ownership, parameter groups, factories
    schedules.py
    precision.py
    state.py
  model/
    lazy.py
    placement.py
    recipes.py
    graph.py
  planning/
    plan.py
    capabilities.py
    partition.py
    cost.py
  backends/torch/
    local.py
    data_parallel.py
    pipeline.py
    sharded.py
    tensor_parallel.py
  data/
    pipeline.py
    sources.py
    sampling.py
  transforms/
    functions.py
    axes.py
  cluster/
    client.py
    coordinator.py
    worker.py
    jobs.py
    protocol.proto
    identity.py
    artifacts.py
    events.py
  checkpoint/
    snapshot.py
    manifest.py
    storage.py
    torch_adapter.py
  providers/
    aws.py
  recipes/
    rl.py
  inference/
    artifact.py
    serving.py
  cli/
    main.py
tests/
  unit/
  numerical/
  integration/
  distributed/
  fault_injection/
examples/
benchmarks/
docs/adr/
```

The engine depends on narrow execution and state interfaces. It does not import AWS or cluster networking. The coordinator invokes job services and never imports the training engine to execute submitted batches. Python/CLI call the same application services. Backend compatibility code remains under `backends/torch`, including any unavoidable use of unstable PyTorch APIs.

Use PyTorch for tensors, autograd, AMP, function transforms, compilation, collectives, and distributed execution. The documented [function transforms](https://docs.pytorch.org/docs/2.15/func.html) supply the basis for MLZO's transform wrappers. Wrap [FSDP2 `fully_shard`](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html) and [tensor parallelism](https://docs.pytorch.org/docs/2.15/distributed.tensor.parallel.html) behind tested strategy adapters rather than exposing those APIs throughout the package.

Use standard-library dataclasses and typing for internal contracts. Proposed optional dependencies are `grpcio`/Protobuf for cluster control, `boto3` for S3/AWS, `safetensors` for portable tensor exports, and Gymnasium for RL recipes. Image helpers may use a compatible torchvision installation. Keep these in extras such as `mlzo[cluster]`, `mlzo[aws]`, `mlzo[vision]`, and `mlzo[rl]`; importing `mlzo` must not require cloud credentials or initialize CUDA.

Choose a tested PyTorch version range during the first milestone. Record exact Python, PyTorch, CUDA, backend-adapter, protocol, and checkpoint-format versions in artifacts. The documentation currently labels pipelining as alpha, so pin its adapter and run compatibility tests before expanding the supported range. See the [pipeline API status](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html).

## 15. Public contracts and internal representations

### 15.1 Core Python contracts

The following describes the proposed typed contract. Supporting types are defined in their owning modules and re-exported only when useful to users. Bodies are intentionally omitted.

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from torch import Tensor, nn

BatchT = TypeVar("BatchT")


@dataclass
class Step:
    loss: Tensor | None = None
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)
    stop: bool | Tensor | None = None
    weight: float | Tensor | None = None


class Training(Generic[BatchT]):
    model: nn.Module
    state: Any | None

    def __init__(self, model: nn.Module) -> None: ...

    def step(self, batch: BatchT) -> Tensor | Step | None: ...

    def update(
        self,
        optimizer: str,
        loss: Tensor,
        *,
        retain_graph: bool = False,
    ) -> None: ...


def train(
    model_or_training: nn.Module | Training[Any] | Callable[[], Training[Any]],
    data: Any | None = None,
    *,
    loss: str | Callable[[Tensor, Tensor], Tensor] | None = None,
    step: Callable[..., Tensor | Step | None] | None = None,
    optimizer: "OptimizerInput" = "adamw",
    lr: float | None = None,
    schedule: "ScheduleInput | None" = None,
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto",
    matmul: Literal["auto", "strict", "fast"] = "auto",
    grad_clip: float | None = None,
    accumulate: int = 1,
    compile: bool | Literal["auto"] = False,
    memory: Literal["auto"] | None = None,
    checkpoint: "CheckpointPolicy | None" = None,
    device: str | None = None,
    seed: int | None = None,
) -> "Run": ...


class Run:
    id: str

    def plan(
        self,
        *,
        on: "Cluster | None" = None,
        parallel: "ParallelInput" = "auto",
        example: Any | None = None,
    ) -> "Plan": ...

    def fit(
        self,
        *,
        epochs: int | None = None,
        steps: int | None = None,
        on: "Cluster | None" = None,
        parallel: "ParallelInput" = "auto",
        code: str | None = None,
        resume: str | None = None,
        wait: bool = True,
    ) -> "History | Run": ...

    def evaluate(
        self,
        data: Any,
        *,
        step: Callable[..., Tensor | Step] | None = None,
    ) -> dict[str, float]: ...
    def checkpoint(self, *, to: str | None = None) -> "CheckpointRef": ...
    def save(self, path: str) -> "CheckpointRef": ...
    def export(self, *, block: str | None = None, to: str) -> "ArtifactRef": ...
    def status(self) -> "RunStatus": ...
    def explain(self) -> str: ...
```

Proposed refinements relative to the initial discussion are `device`, `seed`, `example`, `wait`, an importable training factory, and reference return values. `lr=None` acts as an omission sentinel: built-in optimizers resolve it to `3e-4`, while an existing optimizer/factory retains its own setting. Passing a conflicting explicit `lr` with an optimizer instance raises an error instead of silently changing it.

Use overloads to distinguish models from `Training` and factories, and preserve batch types where practical. A `Training` object cannot also receive a separate `loss` or `step`. `data=None` is valid only for an algorithm with its own iteration source, such as an RL recipe.

`fit()` accepts either `epochs` or `steps`. Here `steps` means completed calls to the learning algorithm, including an RL collection-only iteration. These are distinct from optimizer updates and pipeline microbatches. Limits are total targets within the run; resuming preserves progress toward the target. Increasing a completed run's target is allowed, but changing a schedule horizon requires a recorded reconfiguration.

`wait=True` returns `History` on completion and raises a structured run error on failure. `wait=False` returns the submitted `Run` after durable acceptance. `save(path)` is a convenience for a resumable `checkpoint(to=path)`; `export` produces an inference or block artifact.

Keep pipeline microbatch count in `PipelineConfig`. Do not expose a second competing count on `fit()` in the first release. Likewise, use one serialized optimizer specification with a stable parameter selector rather than several overlapping configuration mechanisms.

### 15.2 The internal learning-program description

Represent the program as immutable specifications plus worker-owned mutable state:

| Internal record | Required contents |
|---|---|
| `RunSpec` | Algorithm kind, model/data recipes, loss/step references, optimizer/schedule specs, requested budgets and policies |
| `FactoryRef` | Artifact digest, importable module, qualified name, validated constructor arguments |
| `ModuleSpec` | Canonical block path, factory reference, aliases, placement constraints, available shape metadata |
| `OptimizerSpec` | Name/factory, selected canonical parameter paths, hyperparameters, update policy |
| `ExecutionPlan` | Run/spec hash, device mesh, stages, tensor edges, ownership, capabilities, normalization and checkpoint rules |
| `JobAssignment` | Run ID, attempt generation, plan digest, rank roles, resource assignment, artifact/storage references |
| `RunState` | Iteration/epoch position, per-optimizer counts, skipped updates, sampler/RNG state, registered algorithm state |
| `CheckpointManifest` | Snapshot identity, generation, progress, version/config identity, logical tensor map, shard checksums |

A `FactoryRef` is an internal transport format. Users continue passing Python classes/functions. Never turn this into a mandatory public model-description language.

The program combines captured model graphs with Python algorithm execution. It does not attempt to compile arbitrary Python training logic into one universal static graph.

### 15.3 Planning has no hidden training side effects

`train()` performs cheap validation only. `plan()` queries metadata and does not provision cloud resources, consume a live data iterator, run optimizer updates, or materialize full weights on the coordinator. If an example or remote probe is required, report that requirement and its validation status.

An isolated worker probe may execute an approved factory in metadata mode or use a bounded representative batch. Its outputs are shape/parameter metadata. Real allocations, code imports, and probe costs must be identified; a probe must not modify the live training instance, sampler, or RNG state.

## 16. Training engine and mathematical semantics

### 16.1 Normalize all customization levels

Convert declarative training to an internal algorithm whose step unpacks the supervised batch, calls the model, and computes the loss. Wrap a functional step in the same interface. Use a user `Training` directly. These adapters then enter the same engine.

The engine has separate iteration and optimizer-update state machines. One iteration may produce zero updates, one accumulated loss contribution, or several explicit updates. A pipeline schedule can contain many microbatches inside one logical forward/loss operation.

Maintain `iteration`, `epoch`, `optimizer_steps[name]`, `skipped_updates[name]`, and `update_events`. One named optimizer update spanning many stage processes counts once globally. `update_events` is the sum of successful named optimizer operations, not a sum over ranks.

### 16.2 Automatic update sequence

For ordinary training:

1. Set the intended training mode and prepare the batch on its consuming device.
2. At the start of an accumulation window, zero the selected optimizer gradients with `set_to_none=True` where supported.
3. Enter the precision and synchronization contexts, then execute forward/loss.
4. Validate that a training loss is scalar and differentiable and that its normalization weight is scalar, finite, nonnegative, and detached. Avoid unnecessarily synchronizing the device for every logged value.
5. Backpropagate the weighted contribution. Release its graph after backward unless the algorithm explicitly needs retention.
6. At the end of the window, synchronize gradients and unscale them when required.
7. Apply the normalization factor, test finiteness, and calculate clipping using the optimizer's complete ownership group.
8. Agree on whether the distributed optimizer operation can proceed. Apply the update or skip it consistently.
9. Advance the relevant counters and scheduler only after a successful operation.
10. Publish detached metrics, honor a stop request, and checkpoint if a safe boundary and policy require it.

PyTorch's AMP examples specify unscaling before clipping and keeping a consistent scale during accumulation. MLZO's precision adapter must preserve those rules. See [AMP examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html).

### 16.3 Loss weighting and accumulation

Let `l_i` be a scalar mean loss and `w_i` its nonnegative, non-differentiable normalization weight. For a window, the desired objective is:

```text
L = sum(w_i * l_i) / sum(w_i)
```

Built-in loss adapters calculate a numerator and denominator using the loss's actual reduction rule. For example, valid-token counts and class-weighted denominators are not interchangeable with batch length. A batch with no valid contributions supplies zero numerator/weight and does not create a NaN mean.

A plain custom scalar has weight 1, so MLZO averages custom step losses equally across a window. A custom mean over variable numbers of examples or tokens supplies `Step(weight=...)` to request sample/token weighting. A custom summed objective can use a single aggregate step; do not silently infer a sum-versus-mean convention from its value.

For each contribution, accumulate the gradient of its numerator and track the denominator. At the update boundary, divide gradients by the actual total weight. This handles a short final window without dividing by the configured `accumulate` count when fewer contributions arrived. A zero-total-weight window clears its gradients and produces no optimizer update.

For DDP with `R` replicas whose backend averages gradients, the accumulated gradients represent `sum_r G_r / R`. After all-reducing the denominator to `W`, multiply by `R / W` to obtain the gradient of the global weighted mean. Local execution uses `1 / W`. This rule supports different valid-token counts across replicas when all replicas still execute a compatible collective schedule.

Pipeline splitting must preserve the same objective. Initially support additive built-in losses and registered loss reducers. Reject automatic microbatch splitting of arbitrary batch-coupled objectives, such as a contrastive loss whose negatives are the whole batch, until an equivalent aggregation policy exists.

Do not divide once in MLZO and again inside a backend pipeline schedule. The pipeline adapter must control the backend's `scale_grads` behavior and test its normalization against the full-batch reference. PyTorch documents that setting on its [pipeline schedules](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html).

### 16.4 Explicit optimizer semantics

`Training.update(name, loss)` synchronously executes a named update through an engine-bound context. Calling it outside an active training step raises an error.

Resolve parameter selectors before execution and reject empty or overlapping sets in the first implementation. Each explicit update clears the selected gradients before backward and clears unused gradients according to ownership afterward. The algorithm remains responsible for graph connectivity, frozen networks, and fresh forwards. The runtime must not freeze another network in a way that prevents gradients flowing through its operations to the selected parameters.

Use separate scaler state per independently updated optimizer in the initial explicit-update mode. A scaler can then advance after that optimizer's transaction, including repeated calls to the same optimizer in one algorithm iteration. Shared parameters, mixed-scale accumulation across optimizers, and shared-graph multi-optimizer updates remain unsupported until separately validated. The standard automatic mode has one scaler per logical optimizer ownership group.

`retain_graph=True` only retains the autograd graph; it does not make it legal to reuse activations after their required parameters have changed. Propagate actionable autograd errors. Returning an automatic loss after explicit updates marks the run failed; already completed explicit updates cannot be rolled back in memory. Recovery uses the last committed checkpoint.

### 16.5 Schedulers, precision, and compilation

Bind optimizer instances after lazy construction and any strategy that changes parameter objects. A scheduler factory runs after its optimizer is created. Preserve native parameter groups and reject incompatible duplicated options.

For built-in warmup, the first upcoming update uses `base_lr / warmup`, and the `warmup`-th successful update reaches `base_lr`. With no warmup, the first update uses `base_lr`. Define the decay endpoints and short-horizon edge cases in tests. Callable schedules receive the upcoming successful-update index and a known horizon and return a multiplier. Unknown horizons require an explicit schedule specification or a horizon-free schedule.

Resolve auto precision once during planning. FP16 overflow decisions must be consistent across all workers participating in the same optimizer operation. For a pipeline, propagate a common loss scale and agree on overflow before any stage updates its parameters. BF16 normally needs no gradient scaler. Restore scaler state on resume.

Calculate a global gradient norm across unique parameter shards. Do not count replicated copies twice. Scope norm and clipping to the selected named optimizer, including all its pipeline stages.

Compile the model or supported stage computation first. Keep orchestration, logging, state mutation, and network control outside compiled functions. `compile=True` reports compilation failures; `compile="auto"` may fall back with a recorded reason. Do not catch an arbitrary failed training call and rerun it, since it may already have mutated state. Restore process-wide numerical settings after local runs and reject conflicting concurrent policies in one process.

### 16.6 Evaluation and metrics

Evaluation uses `eval()` and inference/no-grad execution and restores original module modes, including intentionally frozen submodules. It never calls an explicit-update training method as a proxy for evaluation.

Metrics are detached before storage. Built-in means track numerator and denominator; counts sum; custom metrics declare a reduction when the default mean is inappropriate. Reduce over data replicas, not over pipeline stages that saw the same examples. Throttle device-to-host reads and keep the in-memory history bounded.

## 17. Lazy construction, graph analysis, and parallel execution

### 17.1 Lazy module state machine

Implement `unmaterialized`, `materializing`, `ready`, and `failed` states with protection against repeated or recursive construction. Store the callable and validated arguments without invoking them. A failed construction preserves the original exception and does not silently retry a factory with side effects.

At materialization, resolve the target device, construct on that worker, initialize or restore, validate the result is an `nn.Module`, and register it once. Use a destination-device construction context when supported; constructors with explicit CPU allocations may use worker RAM temporarily. No such fallback allocates weights on the coordinator.

After construction, replace lazy child entries at their original registered module paths with concrete modules. Maintain an alias map when several paths reference the same object. A root `Lazy` resolves to the root execution module. Explicitly materialized wrappers that remain in user containers need a state-dict adapter so MLZO's canonical checkpoint paths do not acquire implementation-only wrapper prefixes.

Raw unresolved wrappers have no wrapped parameters. Unsupported calls to `forward()` before preparation should raise an actionable error rather than accidentally build a remote model on the laptop.

For metadata-supported factories, the backend can use the [PyTorch meta device](https://docs.pytorch.org/docs/2.15/meta.html). Moving an empty/meta module into real storage is not initialization: every parameter and buffer must subsequently be initialized or loaded. Custom constructors with data-dependent work require real worker-side construction or an explicit metadata adapter.

Use a stable hash of run seed and canonical block identity for block initialization. Replicas of the same block receive identical initialization; runtime dropout/data RNG streams are distinct and recorded. Topology-independent seeds do not promise bitwise equality when operations, hardware, or reduction order change.

### 17.2 Planner passes

Implement explicit passes with diagnostics attached to the source module path:

1. Normalize the `RunSpec` and validate algorithm/optimizer combinations.
2. Inspect the module tree, aliases, factories, eager tensors, and placement annotations.
3. Resolve available devices and worker connectivity.
4. Obtain a supported execution graph or explicit stage recipe.
5. Determine parameter ownership and tensor dependencies.
6. Partition computation and assign compatible devices.
7. Resolve precision, compilation, normalization, data sharding, and checkpoint policies.
8. Estimate memory and identify unvalidated shapes or costs.
9. Emit a versioned `ExecutionPlan` and validate it on the assigned workers.

Do not infer execution order solely from `ModuleList` registration order. A model can call modules out of order or reuse them. For supported straight-line Python, use a custom FX tracer that treats unresolved `Lazy` wrappers as leaf calls, then validate its captured operations. Static Python loops can yield a sequence of leaf calls without constructing the layers.

Shape inference for leaf modules runs through metadata adapters or isolated worker probes. A root factory that hides its hierarchy requires worker-side inspection. More general graph capture can use export facilities where compatible. Python control flow depending on tensor values is a known capture boundary, as explained in the [PyTorch export tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html).

Capturing a Python function can execute its Python side effects. Restrict automatic capture to documented supported patterns and never run it against live mutable algorithm state. If capture cannot preserve semantics, require explicit stage construction or use local/data-parallel execution.

### 17.3 Start with single-device remote execution

For one remote GPU, no model partitioner is necessary. A worker receives a root model recipe and data/algorithm configuration, constructs them, and runs the local engine. Do not initialize NCCL merely to run on one GPU.

This provides the user's laptop workflow before multi-GPU planning exists. It also validates packaging, environments, lazy construction, remote storage, metrics, and reconnection independently of pipeline complexity.

### 17.4 Data parallelism

Launch one compute process per assigned GPU, with a separate worker supervisor. Initialize process groups among compute workers only. Wrap compatible modules with DDP and install a distributed sampler or supported source-sharding adapter. DDP synchronizes gradients but does not partition inputs itself; MLZO must provide that part. See [DistributedDataParallel](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html).

For accumulation, place both forward and backward inside `no_sync()` for non-final contributions. Use deterministic epoch shuffling and an explicit uneven-input policy. Initially require aligned numbers of iterations; handle padding with masks or reject incompatible inputs rather than deadlocking.

A custom algorithm must invoke wrapped modules through their callable interface. Advanced algorithms that call independently optimized submodules need ownership-aware wrapping of those submodules. Do not require the user to reach into `.module` or bypass synchronization hooks. Validate this route with GAN and recursive fixtures before marking it supported.

### 17.5 Pipeline construction and execution

Begin with a sequential graph, one stage per rank, additive loss, and fixed input shape buckets. Partition registered leaf calls while retaining functional operations between them. Assign embedding operations to the first stage and the output head/loss to the last stage unless the validated graph says otherwise.

For models that cannot use automatic capture, add an advanced `stage_factory` option to `mlzo.pipeline()`. Its importable factory receives stage identity and configuration and constructs only that stage. Stage input/output schemas declare the connections. Do not ask an explicit stage factory to instantiate the entire model and discard unneeded weights.

Wrap stage modules with PyTorch `PipelineStage` and use its GPipe schedule first, followed by 1F1B. The backend provides activation communication and backward scheduling; MLZO supplies recipes, ownership, data, policies, and validation. Keep these calls in one adapter because the [pipeline API](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html) can change.

At runtime:

1. Every stage constructs only its assigned blocks and optimizer parameters.
2. The first stage receives inputs; targets and required metadata reach their consuming stage directly from the worker data source.
3. The schedule sends activations to downstream workers and retains the necessary local autograd state.
4. The last stage computes the appropriately weighted loss.
5. Backward sends activation gradients upstream; each stage computes its local parameter gradients.
6. After all required microbatches and accumulation contributions, workers normalize, test finiteness, clip, and update consistently.
7. Workers publish metrics and reach a checkpoint-safe boundary.

No stage performs its optimizer update while another microbatch still requires that version of its parameters. Pipeline microbatching splits a batch for scheduling; `accumulate` combines successive loss contributions for an optimizer update. They have separate counters and normalization tests.

Start with checked shape/dtype contracts and either supported short batches or explicit padding/masking. Add dynamic shapes only after adapter tests prove the behavior of the supported PyTorch version. Cross-stage residuals require additional graph edges and lifetime management. Initially reject interleaved ownership such as device 0, then device 1, then device 0 if the selected one-stage-per-rank schedule cannot represent it.

### 17.6 Sharding, tensor parallelism, and automatic placement

Add FSDP2 after DDP and checkpoint correctness. Apply sharding before optimizer construction and use canonical state mapping for save/load. Add tensor-parallel policies only for tested layer patterns, with explicit input/output layouts. A device mesh records the separate data, shard, tensor, and pipeline groups.

The first auto planner follows a deterministic rule set: use one GPU when selected capacity is one; honor valid explicit pipeline constraints; otherwise use data parallelism when the model fits per replica. Choose sharding or pipeline execution only after the required capability and memory checks exist. An unknown memory estimate remains unknown until a probe resolves it.

Model memory includes parameter, optimizer, gradient, activation, communication-buffer, and temporary allocation estimates. Activation liveness depends on schedule and microbatch count. Prefer grouping heavily communicating stages on a fast local interconnect. Heterogeneous devices require capability intersection and measured costs; arbitrary GPU/TPU mixed model parallelism is outside the initial design.

## 18. Data, transformations, and named-axis implementation

Represent `mlzo.data()` as a lazy source description plus ordered operations. Construct loaders only after execution placement is known. Cache keys include source version, transform configuration/code digest, and relevant format versions.

For data parallel execution, shard before expensive preprocessing. Preserve sample IDs through shuffling, padding, recursive carry, and collation. For pipelines, a replica's input stage reads its shard and distributes only required batch fields. Never have every pipeline stage independently iterate the same dataset.

Implement resumable sampling around the consumed sequence position. Prefetched records must either be reproducible from deterministic sample/epoch seeds or be represented in the checkpoint. Native loaders without a state adapter can guarantee only documented boundaries, such as epoch restart. Streaming sources report whether they support exact cursor restore, replay from an offset, or no resume.

Use pinned worker-host memory and asynchronous transfers where supported. `.map()` preserves arbitrary local Python transforms; remote transforms require importable packaged code. GPU transforms execute on workers and belong in the plan's memory/compute accounting. The implementation should follow the separation of datasets, samplers, and workers in [PyTorch data loading](https://docs.pytorch.org/docs/2.15/data.html).

Function transforms are forwarding wrappers with typed signatures and supported keyword arguments, including `argnums`, `has_aux`, batching dimensions, and randomness policies where the backend provides them. Test transform composition and module functionalization; do not silently rewrite side-effectful code.

Implement named axes as `(raw_tensor, axis_names)` with methods that call ordinary PyTorch operations. `align` validates a permutation, `flatten` permutes selected dimensions before flattening, and `mean` resolves axis names to dimensions. Binary operations align names explicitly and reject ambiguous combinations. Wrapping an existing tensor must retain its autograd history rather than copy it through a detached constructor.

## 19. Coordinator, worker service, and remote job protocol

### 19.1 Separate control traffic from tensor traffic

Use an authenticated TLS control service with typed requests, responses, and streamed events. gRPC/Protobuf is the proposed initial choice because it supports generated Python clients and streaming without defining a bespoke wire protocol. Its [Python documentation](https://grpc.io/docs/languages/python/basics/) describes those capabilities, and its [authentication guide](https://grpc.io/docs/guides/auth/) covers transport and call credentials.

The control service handles enrollment, inventory, assignments, cancellation, status, checkpoint requests, and recovery decisions. It has bounded message sizes and no operation for relaying model activations or full parameter sets between workers. Metrics contain detached small values; explicit tensor/artifact downloads use the artifact service separately.

Use PyTorch collectives and point-to-point communication for training tensors. CUDA workers normally use NCCL on a validated network; CPU integration tests can use Gloo. The coordinator has no compute rank and never joins a CUDA process group. Rendezvous can run on a designated worker or persistent service so already-running workers are independent of the laptop.

### 19.2 Worker processes

Each machine runs one long-lived supervisor and isolated job processes. A compute process owns one GPU in the initial distributed design. The supervisor owns enrollment credentials, heartbeat reporting, job preparation, process exit handling, log collection, resource cleanup, and local artifact caches.

Training code runs in the job environment, not inside the supervisor. A training exception must not crash enrollment or corrupt another run's environment. A job identity permits access only to its assigned artifact and storage resources; it does not receive cluster-admin credentials.

Use a small coordinator journal, initially SQLite on persistent local storage, for worker records, invitation usage, run specifications, attempt generations, artifact references, commands, and events. One coordinator owns scheduling authority initially. This is a recoverable single service, not a claim of high availability.

### 19.3 Initial protocol

Define and version these logical operations before implementing Python and CLI clients:

| Operation | Request contents | Result or event |
|---|---|---|
| `EnrollWorker` | Invitation, enrollment request ID, worker public metadata | Scoped worker identity and cluster configuration |
| `Heartbeat` | Identity, sequence, resource inventory, active assignments | Acknowledgment and pending control commands |
| `SubmitRun` | Idempotency key, `RunSpec`, artifact/data digests | Durable run ID and accepted/queued state |
| `PrepareJob` | Run ID, generation, plan digest, rank/device assignment | Prepared state or structured compatibility failure |
| `StartJob` | Prepared assignment and generation | Running event |
| `ReportEvents` | Run/generation, monotonic sequence, bounded event batch | Last persisted sequence |
| `RequestCheckpoint` | Run/generation, request ID, destination policy | Snapshot committed or failed event |
| `CancelJob` | Run/generation, request ID, graceful or abort mode | Final state after worker acknowledgment |
| `DrainWorker` | Worker ID, request ID, drain policy | Draining/idle status |
| `GetRun` | Run ID and event cursor | Current state and events after the cursor |

All mutating operations carry idempotency keys and enforce generation checks. Duplicate delivery cannot start two independent copies of the same attempt. A command acknowledgment means receipt; a separate completion event reports that the requested action finished.

Use bounded exponential backoff with jitter for control reconnection. Give RPCs deadlines and record retryability in errors. Do not blindly retry a model update after losing its acknowledgment.

### 19.4 One-GPU submission workflow

The full happy path is:

1. The local `Run` validates its recipes and resolves the selected cluster.
2. The client snapshots code and declared local datasets and uploads missing content-addressed artifacts.
3. The coordinator journals the submission and reserves one worker GPU.
4. The worker prepares a compatible environment and verifies artifact hashes.
5. It constructs the root model/training/data objects, restores state if requested, and creates the optimizer.
6. The worker records the accepted assignment and begins the common engine.
7. Status and metrics stream back while checkpoint/model tensors stay on the worker/storage path.
8. The worker records final state and releases the job process and GPU reservation.
9. The local handle receives completion or reconnects later to retrieve it.

The accepted assignment includes a finite budget and execution deadline. The supervisor can continue that assignment during a temporary coordinator outage, retain its journal, and checkpoint through an already delegated writer role. A later coordinator must reconcile the active generation before creating a replacement attempt.

### 19.5 Enrollment, trust, and network reachability

Invitations contain cryptographically random enrollment secrets with expiration and a maximum join count. Validate and consume them atomically. Store secret hashes, redact logs, and use an enrollment request ID so a network retry does not consume another join.

`worker.join(token)` must discover a trusted coordinator endpoint. A versioned invitation envelope can include non-secret endpoint/trust information alongside the opaque enrollment credential. Base64 encoding is only representation, never encryption or proof of identity. After enrollment, replace the invitation with a revocable worker-scoped credential; require a separate submitter identity for job execution.

Authenticate TLS servers before submitting credentials. Support scoped call credentials first and certificate-based worker identity where required. Encrypt sensitive data-network traffic through a validated secure fabric or network tunnel; control-plane TLS does not itself encrypt NCCL traffic.

A laptop behind NAT still needs a reachable control route. Initial supported configurations use a reachable endpoint or a configured private overlay network. A later managed discovery/control relay can simplify this without relaying training tensors. Worker-to-worker connectivity is independently required for multi-machine training. Test routes, ports, bandwidth, and backend compatibility before starting the model.

A single remote GPU needs a control connection and storage access; it does not need peer-GPU connectivity. Do not reject that useful case because a multi-worker network is unavailable.

### 19.6 Laptop reconnection and coordinator ownership

Persist run IDs and event cursors locally. If the laptop sleeps, compute workers continue their accepted assignment and spool bounded event logs to persistent worker/storage locations. On reconnect, the client requests events after its last acknowledged cursor and reconciles status.

If the laptop hosted the coordinator, it reloads the journal and reconciles worker generations before scheduling anything. It cannot declare a disconnected attempt dead and launch another writer merely because a heartbeat expired. Fence old attempts or stop for operator-visible recovery when ownership is uncertain.

Changing coordinator ownership requires an explicit transfer of its journal and scheduling authority. The initial release supports restart of the same coordinator and a persistent remote-coordinator option. Automatic leader election and coordinator failover remain separate later work.

## 20. Project packaging and worker environments

### 20.1 Artifact preparation

Use an immutable artifact identified by a digest of its manifest and contents. The manifest records project files, build metadata, declared dependencies, platform requirements, entry-point/factory references, and code identity. A Git commit is useful provenance, but the digest must also cover included uncommitted changes.

For a package with `pyproject.toml`, build a wheel/source artifact using standard packaging conventions. For a small script project, package a bounded source tree with importable modules and explicit dependency metadata. The approach follows [Python packaging guidance](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Snapshot project files into a staging area before hashing/building so edits made during upload cannot produce a mixed version. Exclude `.git`, virtual environments, secrets, caches, model outputs, and data unless declared. Show the included paths, total size, and digest in the run plan. Upload only missing content to the selected artifact store or worker cache.

### 20.2 Cross-platform dependency resolution

A MacBook environment cannot be copied verbatim onto a Linux CUDA machine. Resolve or build dependencies for the worker's operating system, architecture, Python ABI, and accelerator stack. Install the tested PyTorch/CUDA combination from the worker runtime image; do not reuse macOS wheels.

Cache environments by the dependency-lock digest and worker compatibility tuple. Verify package hashes and preserve lock metadata. Native extensions build in a suitable worker/build environment and fail during preparation if unsupported. Dependency preparation must not mutate a shared environment used by another job.

The first worker implementation can use isolated virtual environments for trusted single-owner machines. Managed cloud images can provide container isolation and a preinstalled driver/runtime. Avoid requiring Kubernetes or a container orchestrator for one remote GPU.

### 20.3 Safe reconstruction of Python objects

Automatically encode supported values: primitives, sequences/mappings, approved dataclasses, importable callables, tensor artifact references, MLZO specs, and native objects with registered adapters. Reject closures, open file handles, live generators, arbitrary optimizer instances, and unsupported object graphs with the exact offending path.

For ordinary remote training, a root `Lazy` plus a data descriptor and importable loss/step is sufficient. A custom `Training` instance is not automatically reconstructible just because its class is importable. Support an importable zero-argument training factory through the overload in section 15:

```python
# project/algorithms.py
def make_training():
    model = mlzo.Lazy(make_model)
    return RecursiveTraining(model, cycles=8)


# The local submission code passes the factory without executing it.
run = mlzo.train(make_training, data, optimizer="adamw")
run.fit(epochs=10, on=cluster, code="./")
```

For automatic partitioning, transport a validated graph/spec and assigned module factories. Rebuild stage modules on workers from that description; do not pickle an arbitrary local `GraphModule` or training instance. Existing eager model weights may be uploaded once through a supported adapter, but that path cannot claim the coordinator never held weights.

The worker imports only the pinned project artifact and resolves the referenced callable. Importing approved Python code is code execution and is allowed only for authorized submitters. An invitation to join a worker does not grant permission to submit jobs or inspect another run's artifacts.

## 21. Checkpoint protocol and failure recovery

### 21.1 State boundaries and checkpoint cadence

Checkpoint only at a consistent boundary: all required optimizer operations for the iteration have finished, pipeline work has drained, and an automatic accumulation window has no pending gradients. The initial format does not attempt to save arbitrary in-flight autograd graphs.

`every=N` counts global `update_events`. If a step contains several explicit updates, crossing the threshold schedules a snapshot at that step's safe end. The snapshot records the actual per-optimizer counters; it does not claim to represent exactly update N if an indivisible algorithm iteration crossed it.

An explicit checkpoint request during execution waits for the next safe boundary. Zero-update RL collection can still be checkpointed explicitly at an iteration boundary. Time-based or iteration-based automatic policies can be added without changing the meaning of `every`.

### 21.2 Logical schema

An illustrative committed manifest is:

```json
{
  "format_version": 1,
  "run_id": "run-01",
  "snapshot_id": "snapshot-00017",
  "generation": 2,
  "status": "complete",
  "iteration": 2000,
  "optimizer_steps": {"default": 1000},
  "artifact_digest": "sha256:<project-digest>",
  "config_digest": "sha256:<config-digest>",
  "plan_digest": "sha256:<plan-digest>",
  "versions": {"mlzo": "<version>", "torch": "<tested-version>"},
  "state_index": "state/index.json",
  "shards": [
    {
      "block": "blocks.4",
      "key": "blocks/blocks.4/part-00000.bin",
      "sha256": "<content-digest>",
      "size_bytes": 123456
    }
  ]
}
```

The full schema also includes tensor names/shapes/dtypes and shard ranges, parameter aliases, optimizer group identity, scheduler/scaler state references, RNG ownership, data cursors, custom-state adapters, and an explicit resume guarantee. Tensor identity uses logical parameter names, never just local optimizer integer IDs or GPU numbers.

### 21.3 Commit procedure

1. Allocate a unique snapshot ID and associate it with the current run generation.
2. Reach the safe boundary and agree on the progress counters.
3. Each worker captures an immutable view/copy of its assigned state.
4. Workers write shards directly to the destination and return sizes and checksums.
5. The designated commit writer verifies the expected shard set and state indexes.
6. Publish the immutable complete manifest only when every required shard is durable.
7. Update the `latest` reference conditionally and record completion.
8. Apply retention after the new snapshot is recoverable.

For the first version, block training during snapshot capture and upload. Later asynchronous writing requires stable CPU staging buffers or another snapshot mechanism; background writes must not read live parameters while optimizers mutate them. Bound staging memory and upload concurrency.

The commit writer may be the coordinator or a designated worker driver with scoped generation authority. Worker-owned commit authority allows checkpoints during a laptop disconnect. Only one writer holds that role for a run generation.

Keep the active writer generation and latest snapshot reference in the same authoritative conditional-write record. A replacement attempt first advances that generation through compare-and-swap. An old writer must reject a generation mismatch even if it can read a newer ETag; it must never retry its stale commit against the new generation. Unreferenced old-generation manifests cannot become the current recovery point.

On a local filesystem, use temporary files, durability operations, and atomic replacement where the filesystem supports them. On S3, use immutable keys and conditional writes: create the manifest with `If-None-Match`, and advance `latest` with `If-Match` against its previous ETag. S3 documents these [conditional-write mechanisms](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html). ETags are concurrency tokens, not universal content checksums; record a separate checksum.

An S3 snapshot is not committed by renaming a directory. Strong object consistency does not provide a multi-object transaction, as the [S3 consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel) explains. Failed attempts leave unreferenced objects for cleanup and cannot replace `latest` with a partial snapshot.

### 21.4 Storage and PyTorch integration

Implement a storage interface for immutable object writes, reads/ranges, metadata checks, conditional pointer updates, listing, and deletion. Add local storage first and an S3 adapter using scoped worker credentials or instance roles.

Use PyTorch distributed checkpoint state mappings and its `StorageReader`/`StorageWriter` extension points for sharded state. Confirm supported resharding through actual tests rather than assuming a URL passed to `torch.save` provides distributed storage. PyTorch documents both local-shard saving and version-compatibility limitations in [distributed checkpointing](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html).

Use safetensors for suitable portable weight exports, with a separate alias map for tied parameters. The format stores tensors; it does not serialize the training program or arbitrary Python objects. See [safetensors documentation](https://huggingface.co/docs/safetensors/index).

Store structured state through versioned adapters. Load trusted internal backend checkpoints through the documented restricted loader where possible. Never automatically unpickle arbitrary external objects. Validate schemas, sizes, expected tensor keys, and adapter versions before allocating/restoring.

### 21.5 Resume sequence

1. Resolve a complete manifest and verify version/code/config compatibility.
2. Reconstruct the model recipes and validate the requested topology.
3. Materialize parameters, apply the supported distribution strategy, and construct optimizers against the final parameter objects.
4. Load model/buffer state and canonical optimizer mappings using the selected backend adapter. If loading replaces parameters, rebind optimizers before loading their state.
5. Restore scheduler/scaler/counters, data cursors, and registered algorithm state.
6. Restore runtime RNG after constructors, environment preparation, and any compile/probe work that could consume it.
7. Verify all ranks report the same restored snapshot/generation and continue from the next unconsumed input.

Record the guarantee as exact within a tested deterministic setup, equivalent within numerical tolerance, or partial when an external source/environment cannot restore state. A topology change can preserve weights and optimizer state while changing RNG/reduction order; it is not automatically a bitwise continuation.

### 21.6 Worker and network failure

On a compute-worker failure, stop the affected worker group, abort outstanding collectives, and fence the attempt generation. Do not continue a pipeline with a missing stage or replay just one rank's optimizer step.

Workers may have applied different amounts of in-memory work before a crash. A barrier before updates is not a distributed transaction that can roll those changes back. The recovery unit is the affected group restored from the last committed checkpoint. Recreate communicators, assignments, and optimizers as needed, then resume in a new generation.

Graceful draining requests a safe boundary and snapshot before relinquishing ownership. Scaling a running job initially uses checkpoint, stop, replan, and restore. A later elastic strategy may reuse [PyTorch Distributed Elastic](https://docs.pytorch.org/docs/2.15/distributed.elastic.html), but elasticity still requires an algorithm/data policy and a compatible checkpoint.

Storage failures retry uploads with immutable IDs and preserve the previous committed snapshot. Missing/corrupt shards make a snapshot invalid. Coordinator outages preserve accepted bounded jobs but suspend topology changes until ownership can be reconciled. OOM errors report the stage and observed memory; automatic retry may only use an explicitly allowed equivalent policy.

## 22. Cloud provisioning and resource ownership

Implement AWS after the remote-worker workflow is useful on existing machines. Use the provider SDK through a narrow adapter with `plan`, `create`, `inspect`, `resize`, and `destroy` operations. Reuse the established worker supervisor and enrollment protocol rather than writing an AWS-specific training runtime.

Provisioning resolves the requested GPU model/count into available instance configurations, checks quotas and region support, and presents estimated compute/storage/network costs. Prices and availability are live inputs. `max_hourly_usd` must compare against the allocated instances, including any excess capacity; it is not a guarantee that egress and other charges cannot occur.

Use idempotent provider requests and a durable resource journal. For EC2, the SDK exposes `ClientToken` on [RunInstances](https://docs.aws.amazon.com/boto3/latest/reference/services/ec2/client/run_instances.html). Tag every owned resource with cluster/run ownership and record its provider ID before moving to the next provisioning phase.

The adapter creates or uses an approved network, selects a tested GPU image, assigns least-privilege storage identity, and starts the worker through image bootstrap/user data. A worker enrolls with a short-lived token or provider identity exchange. Training submission uses the worker API, with no SSH dependency.

If provisioning partly fails, reconcile the journal and remove only resources MLZO owns. Existing VPCs, buckets, and externally enrolled hosts remain external resources. Cleanup is idempotent and reports anything it could not release. The same journal supports recovery after a coordinator restart.

Spot interruption requests draining/checkpointing when time permits, but the runtime must also recover from immediate loss. Idle shutdown and maximum runtime are explicit policies. Initial scale operations provision capacity for future jobs or trigger the safe checkpoint/restart path for a compatible active job.

## 23. Export and deployment implementation

First implement a versioned inference artifact with a stable manifest. Store tensor files, construction/export metadata, preprocessing references, schema examples, dependency identity, and an optional validation dataset fingerprint. Keep resumable training checkpoints and inference artifacts distinct in both their types and manifests.

`Run.export` gathers logical model state through worker/storage adapters. Large remote weights stay on the worker-to-storage path. Exporting to a local path explicitly requests downloading the artifact to that machine; the default remote destination does not.

An eager inference artifact contains an importable construction recipe and weights. For compatible models, optionally produce a `torch.export` artifact and record shape constraints and backend requirements. Graph export is an optional compatibility path, not a requirement for ordinary Python inference.

The first deployment worker:

1. Downloads and verifies the selected immutable artifact.
2. Prepares the pinned environment and constructs or loads the model.
3. Moves the model to the selected worker device and selects inference mode.
4. Validates representative inputs, output schema, and memory limits.
5. Warms the model and reports readiness.
6. Serves an authenticated prediction endpoint with bounded queues, deadlines, and validated request sizes.
7. Reports metrics and supports draining and shutdown through the existing control service.

Reuse gRPC for the initial prediction service if an HTTP gateway is unnecessary. Prediction payloads go directly to serving workers or a dedicated inference gateway, never through the laptop's cluster coordinator by default.

Add dynamic batching only for compatible requests and preserve request/output ordering. Deploy a new artifact beside the old version, route traffic after readiness, and retain the previous version for rollback. Distributed inference, tensor-parallel serving, caching for autoregressive generation, quantization, and autoscaling require later capability tests and separate policies.

## 24. Implementation milestones

Each milestone must produce an independently useful result and pass its acceptance checks before its capability becomes a default. These milestones revise the original ordering to deliver the one-remote-GPU laptop workflow early. They specify work to do later; this task only creates the plan.

### M0. Contracts and focused feasibility checks

Implement the package skeleton, typed core contracts, documentation examples, and a tested dependency lock. Establish the public/private module boundary and checkpoint/protocol version fields.

Before expanding infrastructure, run focused experiments for deferred constructor calls, FX leaf capture of a lazy sequential model, worker-side parameter creation, a basic backend pipeline, and a consistent shard snapshot. These answer whether the proposed adapters can meet the zero-weight coordinator requirement.

Acceptance:

- Importing `mlzo` does not initialize CUDA, start networking, or import optional cloud integrations.
- A small classifier has reference PyTorch fixtures for outputs, gradients, optimizer state, and resume.
- The experiments record concrete supported patterns and failures in `docs/adr/`.
- No prototype is promoted into a universal graph-capture or remote-serialization claim.

### M1. A correct local training engine

Implement `train`, `Training`, `Step`, `Run`, declarative/function/class adapters, optimizer factories/groups, explicit named updates, schedules, precision, clipping, accumulation, evaluation, metrics, and local checkpoints. Implement thin function transforms early because they reuse the same backend and have little infrastructure dependency.

Acceptance:

- The three training styles produce equivalent parameter updates for the same classifier.
- Unequal accumulation contributions and a short final window match their mathematical reference.
- A GAN performs fresh discriminator/generator updates in the intended order.
- FP16 overflow skips the corresponding scheduler advance; BF16/FP32 paths remain correct.
- Local checkpoint/resume restores counters, optimizer state, RNG, and supported input position.
- Invalid combinations raise specific errors before the first update when statically detectable.

### M2. Lazy construction, recipes, and local placement

Implement `Lazy`, `place`, canonical block identities, aliases, factory metadata, deferred optimizer binding, and a local `Plan`. Add the data-source description used for remote reconstruction.

Acceptance:

- A model with large declared layers can be constructed without calling their factories or allocating parameter storage.
- Repeated materialization does not duplicate a module or optimizer binding.
- Construction and checkpoint restoration preserve custom initialization rules and tied identities within supported local layouts.
- Placement validation does not call factories or move weights.
- Existing eager PyTorch models still train with the same API.

### M3. MacBook to one remote GPU, without SSH

Implement coordinator journaling, authenticated enrollment, the worker supervisor, project artifacts, worker environment preparation, remote data loading/staging, run submission, events, reconnection, and remote checkpoint storage. Support a root lazy model and importable step/training factories before distributed graph partitioning.

Acceptance:

- A script launched on macOS trains on one Linux CUDA worker with no training computation on the MacBook.
- No SSH or manual per-run code/dependency installation occurs after worker bootstrap.
- The worker constructs the model and optimizer and reads a remote dataset directly.
- A declared local dataset is staged once and cached on a second identical run.
- Metrics appear locally; checkpoint tensors go directly from the worker to storage.
- A temporary laptop disconnect does not stop an accepted healthy job; reconnection restores status/events.
- An incompatible dependency, missing dataset, expired invitation, or unresolvable factory fails during preparation with a useful diagnostic.

This is the first remote product release. It must not wait for FSDP, pipelines, automatic cloud provisioning, or a dashboard.

### M4. Data parallel execution

Implement process-group setup, DDP adapters, distributed source sharding, global loss normalization, synchronized update decisions, metrics reduction, and coordinated checkpoint boundaries. Preserve the same remote submission flow.

Acceptance:

- A two-GPU reference workload matches the intended single-device global-batch update.
- Records are partitioned across replicas and evaluation padding does not duplicate metrics.
- Accumulation avoids unnecessary gradient communication while preserving the final result.
- The coordinator is absent from all compute process groups.
- Worker failure ends or recovers the group rather than leaving a silent collective hang.

### M5. Lazy sequential pipelines and distributed checkpoints

Implement lazy leaf tracing, graph edges, explicit sequential partitions, stage-local factories/optimizers, GPipe followed by 1F1B, pipeline loss normalization, shared overflow/clipping decisions, and complete per-block distributed snapshots. Include an explicit stage-factory route for unsupported capture.

Acceptance:

- A two-stage model matches the forward, gradient, and update reference.
- A ten-block model is placed across ten GPUs without constructing those blocks on the coordinator.
- Network/process instrumentation shows activations and activation gradients moving between workers without a coordinator relay.
- Prefix/suffix operations, loss placement, and stable block IDs survive partitioning.
- Terminate all compute processes after a committed checkpoint, recreate them, and resume coherently.
- Failed shard uploads and stale-generation writers never publish a new valid `latest` snapshot.

### M6. Custom algorithms and data usability

Validate recursive training, GANs, DQN, actor-critic, and then PPO through the existing engine. Add checkpoint adapters for sample-state maps and replay buffers, custom evaluation, richer image/text pipelines, and the initial named-axis wrapper.

Acceptance:

- Recursive carry follows sample IDs and respects the requested truncation/detach behavior.
- Actual selected HRM/TRM implementations are compared with their own reference update rules.
- RL target calculations distinguish termination and truncation; target networks and replay state restore correctly where supported.
- Explicitly unsupported combinations, such as a dynamic recursive algorithm on an unvalidated pipeline schedule, fail before execution.
- None of these examples requires an algorithm-specific branch in the generic engine.

### M7. Sharding and more capable planning

Add FSDP2, registered tensor-parallel policies, supported data/pipeline combinations, residual edges, measured memory estimates, and bounded placement/profile selection. Expand compile and memory policies only for tested combinations.

Acceptance:

- Sharded optimizer state restores with its canonical parameter mapping.
- Supported topology changes load the same logical weights/state with documented numerical guarantees.
- Shared/tied cross-stage parameters either follow a verified synchronization policy or receive an explicit rejection.
- Auto planning produces a reproducible explanation and never changes the objective or effective batch without an explicit policy.

### M8. Managed cloud lifecycle and recovery

Implement AWS provisioning, reviewable allocation plans, budget limits, resource journals, tested worker images, graceful drain, checkpoint-based resize, and owned-resource cleanup. Add persistent coordinator deployment and hardened restart procedures.

Acceptance:

- A developer provisions one or several GPUs, trains from their laptop, saves artifacts, and releases owned resources through Python.
- Provisioning retries and partial failures do not duplicate instances or orphan tracked resources.
- Externally supplied machines and shared infrastructure are not destroyed by cluster cleanup.
- Fault-injection tests cover worker loss, coordinator restart, spot interruption, storage failure, and ambiguous ownership.

### M9. One-GPU deployment

Implement inference artifacts, `deploy`, model readiness checks, schema validation, direct prediction traffic, observability, draining, and version rollback. This milestone can follow M3 and stable export support; it does not need to wait for advanced parallelism or managed provisioning.

Acceptance:

- A trained model is exported and deployed to an existing remote worker from the laptop.
- Serving loads no model weights on the coordinator and performs no inference there.
- Exported inference outputs match the evaluation reference within the declared precision tolerance.
- Invalid requests are rejected before model execution and a bad replacement never becomes ready.
- A deployment restart reloads the same immutable artifact and rollback selects the previous version.

### M10. Scale validation and later backends

After the earlier gates pass, test increasing cluster sizes, coordinator event volume, artifact distribution, storage concurrency, recovery time, and scheduler behavior. Claims about hundreds or thousands of GPUs require those tests and published conditions.

Investigate additional cloud/HPC providers, TPU/JAX execution, richer named-axis rules, distributed inference, performance-guided planning, and any language extension separately. They must not weaken the PyTorch interoperability and correctness contracts already shipped.

## 25. Acceptance and release gates

### 25.1 Numerical correctness

Maintain plain PyTorch references with identical seeds, data order, optimizer options, precision, and global objectives. Compare outputs, loss numerators/denominators, gradients, updated weights, optimizer tensors, schedules, and several-step trajectories.

Use deterministic small CPU FP64 fixtures for tight checks and separate GPU/mixed-precision tolerance suites. Set tolerances per fixture and numerical policy; do not promise bitwise equivalence across device counts or reduction orders. A throughput improvement cannot compensate for an incorrect update.

Required edge cases include a final short accumulation window, zero valid tokens, unequal valid-token counts across ranks, FP16 overflow on one participating stage, repeated named updates, overlapping optimizer rejection, skipped collection-only steps, scheduler resume, tied weights, non-additive loss rejection, and uneven pipeline shapes.

### 25.2 Remote-workflow tests

Run a macOS-client/Linux-worker test and a coordinator process constrained below the model's full parameter-memory requirement. Instrument lazy factory calls and memory separately: low memory use alone does not prove construction was deferred.

Assert that the coordinator owns no model/optimizer tensors in the fully lazy path, joins no compute process group, and receives no activation payloads. Compare control traffic across workloads with different activation sizes to detect an accidental relay. Do not use throughput alone as evidence of the routing guarantee.

Test one GPU, multiple GPUs on one machine, and multiple worker machines. Test project edits between submissions, cached artifacts, native dependency incompatibility, local data staging, remote credentials, invitation expiration/reuse, no-SSH submission, temporary laptop disconnection, and reconnecting to an existing run.

### 25.3 Checkpoint and recovery tests

Inject failure before and after each snapshot phase: capture, shard write, manifest publication, latest-pointer update, and retention. Corrupt/remove a shard, delay an old writer across a generation change, and restart the coordinator with an accepted job still running.

Verify that only complete compatible snapshots can resume; retention preserves the latest recoverable snapshot; worker replay does not double-count committed progress; and changed topology either follows a tested resharding path or fails explicitly. Treat uncheckpointable external state as a documented limitation in the run's manifest/status.

### 25.4 Performance and developer-effort measurements

Record examples/second or tokens/second, peak worker GPU/CPU memory, coordinator memory, communication volume, utilization, pipeline idle time, compile startup, environment preparation, checkpoint pause, and recovery duration. Compare with an equivalent direct-PyTorch configuration using the same numerical policy and workload.

Separate cold-start compilation/environment costs from steady-state throughput. Establish baselines first, then set regression budgets. No fixed speedup is assumed from adding MLZO.

For developer effort, implement the same classifier, custom loss, GAN, DQN, recursive model, remote single-GPU run, multi-GPU model, and checkpoint restoration in MLZO and reference frameworks. Record required concepts, infrastructure steps, user code, and observed debugging work. A shorter script is insufficient if failures become harder to understand.

### 25.5 CI and release policy

- Every change runs CPU unit/numerical tests, static typing, API-example checks, and schema compatibility tests appropriate to its subsystem.
- GPU jobs validate FP32/BF16/FP16 on supported hardware, with distributed tests on the strategies affected by a change.
- Scheduled or release jobs cover multi-machine execution, macOS-to-Linux submission, fault injection, and larger-scale measurements.
- Backend upgrades run the full adapter matrix before the dependency range expands.
- Examples state the milestone/capabilities they require. An unsupported strategy never silently becomes a different strategy.
- New public configuration must remove a real user burden or resolve a correctness ambiguity. Prefer internal policy over another mandatory callback or lifecycle method.

## 26. Decisions requiring validation

These issues have initial implementation choices above, but their experiments determine how far support can expand:

| Issue | Initial choice | Evidence required to expand support |
|---|---|---|
| Zero-weight inspection of ordinary Python models | Root factory for remote single-GPU; leaf tracing for supported lazy graphs | Constructor/memory instrumentation and capture-equivalence fixtures |
| Arbitrary remote training objects | Importable factory or registered adapter | A documented reconstruction and state schema; no generic object pickling |
| Stateful or dynamic pipeline algorithms | Use local/data-parallel execution until validated | Equivalent control flow, loss, gradients, and update ordering |
| Named optimizer overlap | Reject | A precise shared-gradient/optimizer-state ownership policy |
| Closure-based optimizers such as LBFGS | Reject in the generic one-loss path initially | A reentrant closure contract that controls state, RNG, repeated forward, and distributed execution |
| Tied weights across pipeline stages | Reject unsupported cross-stage sharing | Explicit owner/replica synchronization and checkpoint alias tests |
| Variable pipeline shapes | Validated shapes/buckets and masking | Backend-specific shape, communication, and normalization tests |
| Dataset resume | Supported consumed-position adapters | Proof that prefetching, random transforms, and stream offsets restore coherently |
| Laptop behind NAT | Reachable endpoint or configured secure overlay | A managed control-connectivity flow that keeps tensors off the coordinator |
| Coordinator failure | Persistent journal and worker reconciliation | Fencing/ownership proofs and fault tests before automatic failover |
| Cross-version checkpoints | Pin and record backend versions | Explicit migration tooling and numerical resume tests |
| Automatic memory/performance policy | Conservative capability-driven rules | Repeated profiling and equivalence results for every enabled transformation |

The source specification's required subjects are all represented: three training styles, optimizer/scheduler/AMP management, transformations, data processing, named axes, lazy construction, placement, distributed planning, secure clusters, Python enrollment, cloud provisioning, project distribution, block checkpoints, run lifecycle, RL, recursive algorithms, CLI parity, recovery, and migration from ordinary PyTorch. Deployment and the laptop-to-single-GPU workflow receive explicit implementation milestones here.

The first complete remote demonstration is concrete: launch a local Python script on a MacBook, train on one enrolled GPU without SSH, inspect its metrics, save a checkpoint directly from the worker, reconnect after a laptop interruption, resume the same run, and export its model. Multi-GPU demonstrations then add direct activation/gradient exchange and consistent distributed state without changing that submission workflow.
