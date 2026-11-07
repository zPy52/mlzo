"""Explicit unsupported managed provisioning for the alpha release."""
from typing import Any, NoReturn
from mlzo.errors import UnsupportedFeatureError


def aws(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject provisioning without contacting AWS or creating resources."""
    raise UnsupportedFeatureError(
        'AWS managed provisioning is unavailable in MLZO 0.1.0a1. '
        'Enroll an existing machine with Worker.join(); no cloud resources were created.'
    )


__all__ = ['aws']
