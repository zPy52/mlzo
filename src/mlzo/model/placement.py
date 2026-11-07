"""Placement annotations. These functions never allocate or move tensors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from torch import nn

from mlzo.errors import ConfigurationError


def _gpu_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"GPU IDs must be nonnegative logical integers; got {value!r}.")
    return value


def _set_gpu(module: nn.Module, gpu: int) -> None:
    module._mlzo_gpu = gpu
    target = module.__dict__.get("_mlzo_target")
    if isinstance(target, nn.Module):
        target._mlzo_gpu = gpu


def _explicit_gpu(module: nn.Module) -> int | None:
    target = module.__dict__.get("_mlzo_target")
    if isinstance(target, nn.Module) and "_mlzo_gpu" in target.__dict__:
        return target.__dict__["_mlzo_gpu"]
    return module.__dict__.get("_mlzo_gpu")


def place(
    module: nn.Module, *, gpu: int | None = None, gpus: Iterable[int] | None = None
) -> nn.Module:
    """Annotate one module or its ordered direct children with logical GPU IDs."""
    if not isinstance(module, nn.Module):
        raise ConfigurationError("place() requires a torch.nn.Module.")
    if (gpu is None) == (gpus is None):
        raise ConfigurationError("Pass exactly one of gpu= or gpus= to place().")
    if gpu is not None:
        _set_gpu(module, _gpu_id(gpu))
        return module
    if isinstance(gpus, (str, bytes, dict, set, frozenset)):
        raise ConfigurationError("gpus= must be an ordered iterable of logical GPU IDs.")
    try:
        assignments = tuple(_gpu_id(value) for value in gpus)
    except TypeError as exc:
        raise ConfigurationError("gpus= must be an ordered iterable of logical GPU IDs.") from exc
    children = [(name, child) for name, child in module._modules.items() if child is not None]
    if len(assignments) != len(children):
        raise ConfigurationError(
            f"gpus= has {len(assignments)} IDs for {len(children)} ordered direct children. "
            "Unmaterialized Lazy modules have no known children; annotate their root with gpu=."
        )
    seen: dict[int, tuple[str, int]] = {}
    for (name, child), assigned in zip(children, assignments):
        earlier = seen.get(id(child))
        if earlier is not None and earlier[1] != assigned:
            raise ConfigurationError(
                f"Aliased children {earlier[0]!r} and {name!r} refer to the same module "
                f"but request GPUs {earlier[1]} and {assigned}."
            )
        seen[id(child)] = (name, assigned)
    for (_, child), assigned in zip(children, assignments):
        _set_gpu(child, assigned)
    return module


def module_paths(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Walk registered paths, including aliases, and reject cyclic module graphs."""
    def visit(node: nn.Module, path: str, ancestors: frozenset[int]) -> Iterator[tuple[str, nn.Module]]:
        if id(node) in ancestors:
            raise ConfigurationError(f"Cyclic module registration at {path or '<root>'}.")
        yield path, node
        for name, child in node._modules.items():
            if child is not None:
                yield from visit(child, f"{path}.{name}" if path else name, ancestors | {id(node)})

    yield from visit(module, "", frozenset())


def placement_map(module: nn.Module) -> dict[str, int]:
    """Resolve inherited placement and validate shared module/tensor constraints."""
    placements: dict[str, int] = {}
    module_owners: dict[int, tuple[str, int]] = {}
    tensor_owners: dict[int, tuple[str, int]] = {}
    for path, node in module_paths(module):
        parent = path.rpartition(".")[0]
        inherited = placements.get(parent) if path else None
        explicit = _explicit_gpu(node)
        assigned = _gpu_id(explicit) if explicit is not None else inherited
        if assigned is None:
            continue
        placements[path] = assigned
        previous = module_owners.setdefault(id(node), (path, assigned))
        if previous[1] != assigned:
            raise ConfigurationError(
                f"Aliased module paths {previous[0] or '<root>'!r} and {path or '<root>'!r} "
                f"inherit conflicting GPUs {previous[1]} and {assigned}. "
                "Give the shared module one explicit gpu= assignment."
            )
        for name, tensor in (*node._parameters.items(), *node._buffers.items()):
            if tensor is None:
                continue
            tensor_path = f"{path}.{name}" if path else name
            previous_tensor = tensor_owners.setdefault(id(tensor), (tensor_path, assigned))
            if previous_tensor[1] != assigned:
                raise ConfigurationError(
                    f"Tied tensor {previous_tensor[0]!r} / {tensor_path!r} requests "
                    f"conflicting GPUs {previous_tensor[1]} and {assigned}."
                )
    return placements
