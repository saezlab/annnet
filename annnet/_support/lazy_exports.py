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


def resolve_lazy_export(
    namespace: dict[str, Any],
    name: str,
    *,
    modules: dict[str, str] | None = None,
    attrs: dict[str, tuple[str, str]] | None = None,
    functions: dict[str, tuple[str, str]] | None = None,
    package_name: str | None = None,
) -> Any:
    """Resolve and cache one lazy export in ``namespace``.

    ``modules`` are imported eagerly on first attribute access.
    ``attrs`` are imported and returned directly.
    ``functions`` are exposed as call-time lazy wrappers, which preserves the
    current public surface for package entrypoints without importing optional
    backends immediately.
    """
    if modules and name in modules:
        value = import_module(modules[name])
    elif attrs and name in attrs:
        module_name, attr_name = attrs[name]
        value = getattr(import_module(module_name), attr_name)
    elif functions and name in functions:
        if package_name is None:
            raise ValueError('package_name is required for lazy function exports')
        module_name, attr_name = functions[name]
        value = make_lazy_function(module_name, attr_name, package_name)
    else:
        raise AttributeError(name)

    namespace[name] = value
    return value


def export_dir(namespace: dict[str, Any], exported: list[str] | tuple[str, ...]) -> list[str]:
    """Return a stable ``dir()`` view that includes lazy exports."""
    return sorted(list(namespace.keys()) + list(exported))
