"""Generate public API signatures and CLI help from trusted local source.

Uses the standard library; it never imports MLZO or constructs training objects.
Run from site/ with --check to detect missing exports or stale reference pages.
"""
from __future__ import annotations

import argparse
import ast
import copy
from functools import partial
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site/src/content/docs/reference"
SOURCE = ROOT / "src/mlzo"
VERSION = ast.literal_eval(ast.parse((SOURCE / "_version.py").read_text()).body[0].value)

GROUPS = {
    "training": ("Training and runs", "guides/training", {
        "training/run.py": "train Run", "training/types.py": "Training Step History RunStatus",
    }),
    "models": ("Models and factories", "guides/models-and-planning", {
        "model/lazy.py": "Lazy materialize", "model/placement.py": "place", "model/factory.py": "FactoryRef",
    }),
    "data": ("Data pipelines", "guides/data", {"data/pipeline.py": "data images DataPipeline"}),
    "optimization": ("Optimizers and schedules", "guides/optimizers", {
        "training/optim.py": "opt OptimizerSpec", "training/schedules.py": "ScheduleSpec constant linear cosine exponential",
    }),
    "checkpoints": ("Checkpoints", "guides/checkpoints", {
        "checkpoint/policy.py": "checkpoint CheckpointPolicy", "checkpoint/manifest.py": "CheckpointRef",
        "checkpoint/snapshot.py": "save_snapshot load_snapshot list_snapshots",
    }),
    "inference": ("Inference artifacts", "guides/inference", {
        "inference/artifact.py": "ArtifactRef export_model load_artifact", "inference/deployment.py": "Deployment deploy",
    }),
    "cluster": ("Clusters and workers", "remote/setup", {
        "cluster/client.py": "cluster Cluster Invitation run", "cluster/coordinator.py": "Coordinator",
        "cluster/worker.py": "Worker",
    }),
    "transforms": ("Tensors and transforms", "guides/transforms", {
        "transforms/tensor.py": "Tensor wrap", "transforms/named.py": "tensor NamedTensor",
    }),
    "recipes": ("DQN recipe", "guides/dqn", {"recipes/dqn.py": "dqn DQNTraining"}),
    "errors": ("Exceptions", "help/troubleshooting", {
        "errors.py": "MLZOError ConfigurationError UnsupportedFeatureError TrainingError CheckpointError ClusterError",
    }),
}

NOTES = {
    "train": "Returns a deferred `Run`. Model inputs require data and exactly one of `loss` or `step`. Training instances and importable zero-argument training factories supply their own steps. See the [configuration reference](../../configuration/) for every option, default, and validation rule.",
    "Run": "Use `mlzo.train()` to create a configured run. `fit()` returns `History` locally or after a waited remote run; asynchronous remote submission returns the handle. Budgets are total targets. Remote resume and evaluation are unsupported.",
    "Run.plan": "Returns a metadata plan with `explain()` and `to_dict()`. It does not construct lazy models or consume data; remote inventory queries may contact the coordinator.",
    "Run.explain": "Returns the plan explanation with any execution notes available on a prepared run.",
    "Run.wait": "Returns successful terminal history. Raises `TrainingError` if no suitable run is attached or a remote run fails or is cancelled.",
    "Run.status": "Returns `RunStatus` with state, ID, counters, and optional error information.",
    "Run.evaluate": "Returns averaged scalar metrics from local evaluation. Requires an evaluator for custom algorithms and rejects remote evaluation.",
    "Run.checkpoint": "Returns a full checkpoint reference at a safe local boundary or delegates to the completed remote run. A local destination or configured policy is required.",
    "Run.save": "Alias for `checkpoint(to=path)`.",
    "Run.export": "Exports immutable inference weights to a fresh local or assigned-worker directory, optionally selecting a registered block.",
    "Run.cancel": "Requests cancellation of the attached remote job or an active local engine.",
    "Step": "`loss` is a scalar differentiable mean for automatic optimization. `metrics` maps names to scalar values. `stop=True` requests a whole-run stop. `weight` is the finite nonnegative normalization denominator; omitted weight means one.",
    "Training": "Subclass and implement `step`. Only `state` is checkpointed, so use tensors and supported primitive containers. Set `step_driven=True` for collection iterations receiving `None`.",
    "History": "Retains at most `max_records` recent detached scalar records. `records`, `dropped`, `counters`, `state`, and `notes` describe that bounded report.",
    "History.metrics": "Returns retained metric values grouped by metric name.",
    "History.append": "Appends one detached scalar metric record and optional iteration, epoch, and update-event counters. Old records are evicted at the configured bound.",
    "RunStatus": "A report containing `state`, `id`, `counters`, and `error`. Prefer `run.status()` over constructing reports manually.",
    "Lazy": "An `nn.Module` wrapper that records a factory and its arguments. `state` is unmaterialized, materializing, ready, or failed. Forward never triggers construction. A recorded `gpu` value is an annotation, not a tensor move.",
    "Lazy.state": "Returns the construction lifecycle state.",
    "Lazy.error": "Returns the retained construction exception, or `None`.",
    "Lazy.materialize": "Materializes the recipe once on the requested device and returns the concrete module.",
    "Lazy.forward": "Delegates to the concrete module after explicit materialization; raises `TrainingError` while unresolved.",
    "Lazy.train": "Sets the train/eval mode using ordinary PyTorch module semantics and returns this wrapper.",
    "Lazy.state_dict": "Returns the materialized module's canonical state dictionary when ready, preserving parameter names.",
    "Lazy.extra_repr": "Returns PyTorch's extra representation of the stored factory, arguments, state, and GPU annotation.",
    "FactoryRef": "Names a trusted Python callable by importable module and qualified name. Rejects `__main__`, closures, and references that do not resolve consistently.",
    "FactoryRef.resolve": "Imports the module and returns the named callable. This executes trusted Python module code.",
    "FactoryRef.from_callable": "Validates that a named callable can be reconstructed by its importable reference.",
    "OptimizerSpec": "Deferred optimizer name or factory, parameter selectors, and keyword options. Construct conveniently with `opt()`.",
    "ScheduleSpec": "Validated built-in schedule configuration. Rates multiply base learning rates and advance on successful updates.",
    "constant": "Returns a constant multiplier specification with optional update-indexed warmup.",
    "linear": "Returns a linear decay specification. An omitted horizon must be inferable for automatic training.",
    "cosine": "Returns a cosine decay specification with `min_factor` as the final base-rate multiplier.",
    "exponential": "Returns geometric decay per successful optimizer update after optional warmup.",
    "DataPipeline": "Immutable ordered pipeline configuration. Construction and serialization do not call sample functions or read data. A build opens sources; iteration requests samples. Source objects remain caller-owned.",
    "ArtifactRef": "A local immutable inference artifact reference. It is distinct from a resumable training checkpoint.",
    "ArtifactRef.artifact_id": "Returns the identity recorded in the artifact manifest.",
    "Deployment": "A synchronous predictor in the current process. Use `deploy()` or its context manager to load and release the model.",
    "Deployment.predict": "Unwraps nested MLZO tensor leaves without reshaping, moves native tensors to the configured device, and predicts in eval mode under `torch.inference_mode()`. Raises after shutdown.",
    "Deployment.status": "Reports local readiness and artifact identity.",
    "Deployment.device": "Returns the device used by this local predictor.",
    "Deployment.shutdown": "Releases the model. Repeated shutdown is safe.",
    "Cluster.status": "Queries coordinator status using this client's credential.",
    "Cluster.scale": "Raises `UnsupportedFeatureError`. Managed provisioning and scaling are unavailable; enroll existing machines explicitly.",
    "Cluster.workers": "Returns enrolled-worker membership and liveness reported by the coordinator.",
    "Cluster.invite": "Creates an expiring worker enrollment invitation. Defaults to 900 seconds and one join; requires admin authority.",
    "Cluster.drain": "Stops new assignments to the named worker.",
    "Cluster.remove": "Revokes the named worker's membership.",
    "Cluster.submit": "Snapshots a trusted importable project and returns its durable run ID. One worker executes the run. Remote resume is rejected.",
    "Cluster.get_run": "Returns the durable coordinator record for this run ID.",
    "Cluster.events": "Returns scalar events after the supplied sequence cursor.",
    "Cluster.wait": "Polls until a terminal record or the optional timeout. Returns the record; use `Run.wait()` for History and failure exceptions.",
    "Cluster.cancel": "Requests cancellation of the run.",
    "Cluster.command_status": "Queries a previously submitted worker artifact command by command ID.",
    "Cluster.checkpoint_run": "Returns a completed run's retained reference or requests a checkpoint at a worker-local destination. Does not download it.",
    "Cluster.export_run": "Requests a completed run's export on its assigned worker. The service and job directory must remain available.",
    "Worker.join": "Enrols using an Invitation or encoded invitation, persists scoped identity, and returns a worker. Provide explicit CA trust for private TLS.",
    "Worker.status": "Reports persisted local metadata and this object's service state. Query `Cluster.workers()` for coordinator membership and liveness.",
    "Worker.start": "Starts this worker service in a background thread and returns the worker.",
    "Worker.serve": "Runs the polling service until stopped. One job/device executes at a time and the directory is exclusively locked.",
    "Worker.stop": "Requests this worker object's service stop, optionally allowing current work to finish.",
    "Worker.leave": "Leaves coordinator membership, gracefully by default, and returns the result.",
    "Coordinator.start": "Starts the coordinator in a background thread and returns it. Context-manager entry also starts it.",
    "Coordinator.serve": "Serves in the current thread until shutdown.",
    "Coordinator.shutdown": "Stops this coordinator. `Cluster.shutdown()` instead requests worker service termination.",
    "Coordinator.endpoint": "Returns the advertised origin or the bound endpoint after startup.",
    "Coordinator.token": "Returns the admin secret. Do not print or store it in project source.",
    "Coordinator.submitter_token": "Returns the submitter secret. It is distinct from the admin credential.",
    "Invitation.encode": "Returns the transferable secret invitation envelope. Treat the result as a credential.",
    "Invitation.decode": "Validates an encoded invitation envelope or accepts an existing invitation. It does not establish CA trust.",
    "Tensor": "An explicit lightweight wrapper over native tensor storage and autograd. Only declared methods are supported; use `.raw` for other native operations, modules, optimizers, function transforms, and compiled regions. Basic integer indexing keeps singleton axes; any advanced/boolean component makes the whole index native. Explicit shape operations keep their requested rank.",
    "tensor": "Converts with `torch.as_tensor` and returns `Tensor`, or `NamedTensor` with axes. Default normalization turns scalars into `(1, 1)` and vectors into `(N, 1)`; `normalize=False` preserves input shape. Names must match the final shape, so named vectors need `normalize=False` or two names. Existing storage/history survive when dtype and device permit; detach explicitly to disable an existing graph.",
    "wrap": "Wraps a native tensor without conversion, normalization, copying, or detaching: `wrap(native).raw is native`. An existing MLZO wrapper is returned unchanged.",
    "Tensor.shape": "Returns the underlying tensor's exact shape.",
    "Tensor.ndim": "Returns the underlying tensor's rank.",
    "Tensor.dtype": "Returns the underlying tensor's PyTorch dtype.",
    "Tensor.device": "Returns the underlying tensor's device.",
    "Tensor.requires_grad": "Reports whether autograd tracks the underlying tensor.",
    "Tensor.to": "Uses native PyTorch conversion semantics and wraps the result. Named inputs retain names.",
    "Tensor.astype": "Converts dtype using a torch dtype or its string name, such as `float32`. Named inputs retain names.",
    "Tensor.on": "Transfers to a device using native PyTorch semantics. Named inputs retain names.",
    "Tensor.remove_dim": "Removes only the specified singleton dimension; raises if its size is not one. The result is not normalized.",
    "Tensor.transpose": "Swaps two dimensions, defaulting to the last two. Explicit shape results are not normalized.",
    "Tensor.reorder": "Permutes dimensions in the supplied integer order.",
    "Tensor.flatten": "Flattens the selected positional dimension range using native PyTorch semantics, without result normalization.",
    "Tensor.detach": "Detaches from autograd and wraps the result. Named inputs retain names.",
    "Tensor.clone": "Clones native storage with the supplied PyTorch options. Named inputs retain names.",
    "Tensor.backward": "Delegates differentiation to the native tensor, unwrapping an optional MLZO gradient tensor.",
    "NamedTensor": "A Tensor subclass with one unique nonempty name per dimension. Inherits Tensor conversion and reduction methods. Named arithmetic aligns and broadcasts shared axes, then appends new right-hand names. Unnamed nonscalar operands are rejected. See the guide for flatten placement and operations that drop names.",
    "NamedTensor.transpose": "Swaps dimensions and their axis names, defaulting to the last two dimensions.",
    "NamedTensor.reorder": "Permutes dimensions and their names in the supplied integer order.",
    "NamedTensor.remove_dim": "Removes a singleton dimension and its name without normalizing the result.",
    "NamedTensor.to": "Converts the underlying tensor using PyTorch semantics while retaining names.",
    "NamedTensor.clone": "Clones the underlying tensor and retains axis names.",
    "NamedTensor.detach": "Detaches the underlying tensor from autograd and retains axis names.",
    "NamedTensor.raw": "The ordinary PyTorch tensor. Use it for unsupported wrapper operations.",
    "NamedTensor.shape": "Returns the underlying tensor's shape in axis order.",
    "NamedTensor.ndim": "Returns the number of named dimensions.",
    "NamedTensor.dtype": "Returns the underlying tensor's PyTorch dtype.",
    "NamedTensor.device": "Returns the underlying tensor's execution device.",
    "NamedTensor.requires_grad": "Reports whether autograd tracks the underlying tensor.",
    "DQNTraining": "A step-driven discrete-action algorithm. Construct with `dqn()`. Replay and target state resume, but the environment resets.",
    "DQNTraining.step": "Collects one environment transition and, after replay warmup, attempts a default optimizer update. Returns detached scalar metrics.",
    "DQNTraining.close": "Closes environment resources and prevents further collection. Repeated calls are safe; the engine does not close it automatically.",
}


def signature(node: ast.FunctionDef, name: str) -> str:
    arguments = copy.deepcopy(node.args)
    first = arguments.posonlyargs or arguments.args
    if first and first[0].arg in ("self", "cls"):
        first.pop(0)
    args = ast.unparse(arguments)
    returns = "" if node.name == "__init__" or node.returns is None else " -> " + ast.unparse(node.returns)
    if len(name + args + returns) > 100:
        parts = []
        positional = arguments.posonlyargs + arguments.args
        default_start = len(positional) - len(arguments.defaults)
        for index, argument in enumerate(positional):
            value = ast.unparse(argument)
            if index >= default_start:
                value += " = " + ast.unparse(arguments.defaults[index - default_start])
            parts.append(value)
            if arguments.posonlyargs and index + 1 == len(arguments.posonlyargs):
                parts.append("/")
        if arguments.vararg:
            parts.append("*" + ast.unparse(arguments.vararg))
        elif arguments.kwonlyargs:
            parts.append("*")
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            parts.append(ast.unparse(argument) + (" = " + ast.unparse(default) if default is not None else ""))
        if arguments.kwarg:
            parts.append("**" + ast.unparse(arguments.kwarg))
        return name + "(\n" + "".join(f"    {part},\n" for part in parts) + ")" + returns
    return f"{name}({args}){returns}"


def member_signature(node: ast.FunctionDef, owner: str) -> str:
    if any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list):
        annotation = ": " + ast.unparse(node.returns) if node.returns is not None else ""
        return f"{owner}.{node.name}{annotation}  # read-only property"
    return signature(node, owner + "." + node.name)


def description(node: ast.AST, name: str) -> str:
    if name in NOTES:
        return NOTES[name]
    doc = ast.get_docstring(node)
    if doc:
        return doc.replace("``", "`")
    method = name.rsplit(".", 1)[-1]
    if method == "to_dict":
        return "Returns a dictionary representation. Public invitation and cluster representations omit credentials."
    if method == "from_dict":
        return "Reconstructs this value from its dictionary representation, applying its validation rules."
    raise ValueError(f"Write a description for {name}")


def render_symbol(node: ast.AST, name: str, path: str) -> str:
    source_link = f"https://github.com/zPy52/mlzo/blob/main/src/mlzo/{path}#L{node.lineno}"
    result = f"\n## {name}\n\n"
    if isinstance(node, ast.FunctionDef):
        result += f"```python\n{signature(node, 'mlzo.' + name)}\n```\n\n"
    else:
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init:
            result += f"```python\n{signature(init, 'mlzo.' + name)}\n```\n\n"
        fields = [n for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and not n.target.id.startswith('_')]
        if fields:
            result += "Declared fields:\n\n```python\n" + "\n".join(ast.unparse(n) for n in fields) + "\n```\n\n"
    result += description(node, name) + f"\n\n[View source]({source_link}).\n"
    if isinstance(node, ast.ClassDef):
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                result += f"\n### {name}.{method.name}\n\n```python\n{member_signature(method, name)}\n```\n\n{description(method, name + '.' + method.name)}\n"
    return result


def cli_page() -> str:
    tree = ast.parse((SOURCE / "cli/main.py").read_text())
    # Execute only the trusted parser definition, with a standard argparse shim.
    # No MLZO imports, commands, I/O handlers, or service code run here.
    parser_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parser")

    class Parser(argparse.ArgumentParser):
        def __init__(self, *args, json_output=False, **kwargs):
            super().__init__(*args, **kwargs)

    namespace = {"argparse": argparse, "partial": partial, "_Parser": Parser, "__version__": VERSION}
    exec(compile(ast.Module(body=[parser_fn], type_ignores=[]), "<cli-parser>", "exec"), namespace)
    parser = namespace["_parser"](False)
    page = "---\ntitle: Command-line interface\ndescription: Exact MLZO command help generated from the implemented argument parser.\n---\n\n"
    page += "`mlzo` and `python -m mlzo` are equivalent. `--json` is accepted anywhere and wraps successes as `{\"ok\": true, \"result\": ...}` and errors as `{\"ok\": false, \"error\": ...}`. Services emit a startup record before blocking. Errors return a nonzero exit code and omit sensitive details.\n\n"
    page += "Use [remote setup](../../remote/setup/) for a complete command sequence. Artifact paths returned by remote checkpoint/export commands belong to the assigned worker. `cluster scale` and `run resume` exist to return explicit unsupported errors.\n\n"
    page += "Token input uses protected files or environment variables. Invitation output needs a new `--token-file` path unless you explicitly choose `--show-token`.\n\n"

    def walk(current, level):
        nonlocal page
        page += f"{'#' * level} {current.prog}\n\n```text\n{current.format_help().strip()}\n```\n\n"
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    walk(child, min(level + 1, 4))

    walk(parser, 2)
    return page.rstrip() + "\n"


def main():
    check = "--check" in sys.argv
    generated = {}
    covered = {"__version__", "schedules", "transforms", "worker", "grad", "vmap", "jacrev", "jacfwd", "hessian", "jvp", "vjp", "functional_call", "jit"}
    for group, (title, guide, files) in GROUPS.items():
        page = f"---\ntitle: {title}\ndescription: Public MLZO {VERSION} API signatures and behavior for {title.lower()}.\n---\n\n"
        page += f"Read the [{title.lower()} guide](../../../{guide}/) for examples and usage constraints. Signatures and declared fields below come from MLZO `{VERSION}` source. Type names follow their defining Python modules.\n"
        for file, names in files.items():
            tree = ast.parse((SOURCE / file).read_text())
            definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name in names.split():
                page += render_symbol(definitions[name], name, file)
                covered.add(name)
        if group == "optimization":
            page += "\n## schedules\n\n`mlzo.schedules` is the schedule module. Its `constant`, `linear`, `cosine`, and `exponential` constructors are also exported directly from `mlzo`.\n"
        if group == "transforms":
            page += "\n## PyTorch aliases\n\n`mlzo.transforms` contains the same aliases and tensor helpers exported at the top level. Use native tensor inputs and `.raw` inside transformed functions or compiled regions; wrappers have no transparent torch dispatch.\n\n| MLZO | PyTorch reference |\n| --- | --- |\n"
            for name in "grad vmap jacrev jacfwd hessian jvp vjp functional_call jit".split():
                original = "compile" if name == "jit" else "func." + name
                page += f"| `mlzo.{name}` | [`torch.{original}`](https://docs.pytorch.org/docs/stable/generated/torch.{original}.html) |\n"
        if group == "cluster":
            tree = ast.parse((SOURCE / "cluster/worker.py").read_text())
            join = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "join")
            page += f"\n## worker module\n\n`mlzo.worker` is the worker module. `mlzo.worker.join` delegates to `Worker.join` and returns a worker. The module also exposes the `Worker` class.\n\n```python\n{signature(join, 'mlzo.worker.join')}\n```\n"
        generated[OUT / "api" / (group + ".md")] = page
    top = ast.parse((SOURCE / "__init__.py").read_text())
    exports = next(ast.literal_eval(n.value) for n in top.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets))
    if set(exports) != covered:
        raise ValueError(f"Public export coverage differs: missing={set(exports) - covered}, obsolete={covered - set(exports)}")
    generated[OUT / "cli.md"] = cli_page()
    stale = []
    for path, content in generated.items():
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit("Stale reference pages. Run npm run reference:\n" + "\n".join(stale))
    print(f"{'Checked' if check else 'Generated'} {len(generated)} reference pages covering all {len(exports)} public exports.")


if __name__ == "__main__":
    main()
