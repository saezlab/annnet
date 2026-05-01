"""Shared optional component registry."""

from __future__ import annotations

from typing import NamedTuple
from importlib.util import find_spec
from collections.abc import Mapping


class OptionalComponent(NamedTuple):
    module: str
    install: str | None = None


GRAPH_BACKENDS: dict[str, OptionalComponent] = {
    'networkx': OptionalComponent('networkx', 'annnet[networkx]'),
    'igraph': OptionalComponent('igraph', 'annnet[igraph]'),
    'graph-tool': OptionalComponent('graph_tool', 'pixi / conda-forge / system package'),
    'pyg': OptionalComponent('torch_geometric', 'annnet[pyg]'),
}

PLOT_BACKENDS: dict[str, OptionalComponent] = {
    'graphviz': OptionalComponent('graphviz', 'annnet[graphviz] or annnet[plot]'),
    'pydot': OptionalComponent('pydot', 'annnet[pydot] or annnet[plot]'),
    'matplotlib': OptionalComponent('matplotlib', 'annnet[matplotlib] or annnet[plot]'),
}

DATAFRAME_BACKENDS: dict[str, OptionalComponent] = {
    'polars': OptionalComponent('polars', 'annnet[polars]'),
    'pandas': OptionalComponent('pandas', 'annnet[pandas]'),
    'pyarrow': OptionalComponent('pyarrow', 'annnet[pyarrow]'),
}

IO_MODULES: dict[str, OptionalComponent] = {
    'annnet': OptionalComponent('annnet.io.annnet_format'),
    'json/ndjson': OptionalComponent('annnet.io.json_format'),
    'dataframes': OptionalComponent('annnet.io.dataframes'),
    'csv': OptionalComponent('annnet.io.csv_format'),
    'excel': OptionalComponent('openpyxl', 'annnet[excel]'),
    'graphml/gexf': OptionalComponent('networkx', 'annnet[networkx]'),
    'sif': OptionalComponent('annnet.io.sif'),
    'cx2': OptionalComponent('annnet.io.cx2'),
    'parquet': OptionalComponent('pyarrow', 'annnet[parquet]'),
    'zarr': OptionalComponent('zarr', 'annnet[zarr_io]'),
    'sbml': OptionalComponent('lxml', 'annnet[sbml]'),
    'omnipath': OptionalComponent('annnet.io.omnipath'),
}


def component_names(specs: Mapping[str, OptionalComponent]) -> tuple[str, ...]:
    """Return component names in preference/display order."""
    return tuple(specs)


def _is_component_available(component: OptionalComponent) -> bool:
    """Return whether a component's import target is available."""
    return find_spec(component.module) is not None


def available_optional_components(specs: Mapping[str, OptionalComponent]) -> dict[str, bool]:
    """Return availability by component name."""
    return {name: _is_component_available(component) for name, component in specs.items()}


def component_status(specs: Mapping[str, OptionalComponent]) -> dict[str, dict[str, str]]:
    """Return docs/UI-friendly component status."""
    return {
        name: {
            'available': 'yes' if _is_component_available(component) else 'no',
            'install': component.install or 'built in',
        }
        for name, component in specs.items()
    }


def select_component(
    specs: Mapping[str, OptionalComponent],
    preferred: str | None,
    *,
    kind: str,
    install_message: str,
) -> str:
    """Resolve a component name from a registry."""
    requested = 'auto' if preferred is None else str(preferred).lower()
    names = component_names(specs)

    if requested == 'auto':
        available = available_optional_components(specs)
        for name in names:
            if available[name]:
                return name
        raise RuntimeError(f'No {kind} backend available. {install_message}')

    if requested not in specs:
        allowed = ', '.join(('auto', *names))
        raise ValueError(f'Unknown {kind} backend {preferred!r}; expected one of: {allowed}.')

    if not _is_component_available(specs[requested]):
        raise RuntimeError(
            f'{kind.capitalize()} backend {requested!r} is not installed. '
            f"{install_message}, or use backend='auto'."
        )
    return requested
