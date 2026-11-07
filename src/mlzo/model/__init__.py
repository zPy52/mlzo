"""Deferred module construction, factory recipes, and placement metadata."""

from .factory import FactoryRef
from .lazy import Lazy, LazyState, get_recipe, materialize
from .placement import module_paths, place, placement_map

__all__ = ["FactoryRef", "Lazy", "LazyState", "get_recipe", "materialize", "module_paths", "place", "placement_map"]
