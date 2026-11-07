"""MLZO's Python alpha API.

Import the implementation packages before binding same-named factory exports.
Python then caches those packages, so subsequent submodule imports preserve
the callable ``data``, ``checkpoint``, and ``cluster`` entrypoints.
"""

from ._version import __version__
from .errors import (
    CheckpointError,
    ClusterError,
    ConfigurationError,
    MLZOError,
    TrainingError,
    UnsupportedFeatureError,
)
from .model import FactoryRef, Lazy, materialize, place
from .training import (
    History,
    OptimizerSpec,
    Run,
    RunStatus,
    ScheduleSpec,
    Step,
    Training,
    constant,
    cosine,
    exponential,
    linear,
    opt,
    schedules,
    train,
)
from .data import DataPipeline, data, images
from . import transforms
from .transforms import (
    NamedTensor,
    Tensor,
    functional_call,
    grad,
    hessian,
    jacfwd,
    jacrev,
    jit,
    jvp,
    tensor,
    vjp,
    vmap,
    wrap,
)
from .checkpoint import (
    CheckpointPolicy,
    CheckpointRef,
    checkpoint,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)
from .cluster import Cluster, Coordinator, Invitation, Worker, cluster, run, worker
from .inference import ArtifactRef, Deployment, deploy, export_model, load_artifact
from .recipes import DQNTraining, dqn

__all__ = [
    "__version__", "train", "Training", "Step", "Run", "History", "RunStatus",
    "opt", "OptimizerSpec", "schedules", "ScheduleSpec", "constant", "linear",
    "cosine", "exponential", "Lazy", "materialize", "place", "FactoryRef",
    "data", "DataPipeline", "images", "transforms", "grad", "vmap", "jacrev", "jacfwd",
    "hessian", "jvp", "vjp", "functional_call", "jit", "tensor", "Tensor", "NamedTensor", "wrap",
    "checkpoint", "CheckpointPolicy", "CheckpointRef", "save_snapshot",
    "load_snapshot", "list_snapshots", "cluster", "Cluster", "Invitation",
    "Coordinator", "Worker", "worker", "run", "ArtifactRef", "Deployment", "export_model",
    "load_artifact", "deploy", "MLZOError", "ConfigurationError",
    "UnsupportedFeatureError", "TrainingError", "CheckpointError", "ClusterError",
    "dqn", "DQNTraining",
]
