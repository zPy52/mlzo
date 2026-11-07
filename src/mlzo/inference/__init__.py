"""Immutable local inference artifacts and synchronous local deployment."""

from .artifact import ArtifactRef, export_model, load_artifact
from .deployment import Deployment, deploy

__all__ = ["ArtifactRef", "Deployment", "export_model", "load_artifact", "deploy"]
