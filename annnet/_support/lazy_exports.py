"""Shared helpers for package-level lazy exports."""

from __future__ import annotations

from typing import Any
from importlib import import_module


def make_lazy_function(module_name: str, attr_name: str, package_name: str):
    """Return a call-through wrapper that imports the target on first use."""

    def _lazy_function(*args, **kwargs):
        module = import_module(module_name)
        return getattr(module, attr_name)(*args, **kwargs)

    _lazy_function.__name__ = attr_name
    _lazy_function.__qualname__ = attr_name
    _lazy_function.__module__ = package_name
    return _lazy_function


def load_attr(mapping: dict[str, tuple[str, str]], name: str) -> Any:
    """Load an attribute described by a ``name -> (module, attr)`` mapping."""
    module_name, attr_name = mapping[name]
    return getattr(import_module(module_name), attr_name)


def export_dir(namespace: dict[str, Any], exported: list[str] | tuple[str, ...]) -> list[str]:
    """Return a stable ``dir()`` view that includes lazy exports."""
    return sorted(list(namespace.keys()) + list(exported))
