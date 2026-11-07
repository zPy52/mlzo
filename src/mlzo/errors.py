"""Public exception types shared by MLZO modules."""


class MLZOError(Exception):
    """Base error for an MLZO operation."""


class ConfigurationError(MLZOError, ValueError):
    """A configuration cannot be executed as requested."""


class UnsupportedFeatureError(ConfigurationError, NotImplementedError):
    """A requested capability is outside the supported alpha subset."""


class TrainingError(MLZOError):
    """An accepted training run failed."""


class CheckpointError(MLZOError):
    """A checkpoint could not be saved, verified, or restored."""


class ClusterError(MLZOError):
    """A cluster operation failed."""
