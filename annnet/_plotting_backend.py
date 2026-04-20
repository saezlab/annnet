"""Central plotting backend helpers."""

from __future__ import annotations

from ._optional_components import (
    PLOT_BACKENDS,
    available_optional_components,
    component_names,
    select_component,
)

PLOT_BACKEND_PRIORITY = component_names(PLOT_BACKENDS)
_DEFAULT_PLOT_BACKEND = "auto"


def available_plot_backends() -> dict[str, bool]:
    """Return installed plotting backends in AnnNet preference order."""
    return available_optional_components(PLOT_BACKENDS)


def select_plot_backend(preferred: str | None = "auto") -> str:
    """Resolve a plotting backend name.

    ``"auto"`` selects the first installed backend in this order: Graphviz,
    pydot, then matplotlib.
    """
    preferred = _DEFAULT_PLOT_BACKEND if preferred is None else preferred
    return select_component(
        PLOT_BACKENDS,
        preferred,
        kind="plotting",
        install_message="Install graphviz, pydot, or matplotlib",
    )


def get_default_plot_backend() -> str:
    """Return the configured default plotting backend."""
    return _DEFAULT_PLOT_BACKEND


def set_default_plot_backend(backend: str | None = "auto") -> str:
    """Set the default backend used by ``plot(..., backend=None)``."""
    global _DEFAULT_PLOT_BACKEND

    requested = "auto" if backend is None else str(backend).lower()
    if requested != "auto":
        select_plot_backend(requested)
    elif not any(available_plot_backends().values()):
        select_plot_backend("auto")
    _DEFAULT_PLOT_BACKEND = requested
    return _DEFAULT_PLOT_BACKEND
