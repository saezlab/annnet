"""
These tests don't replace a real static type-check (we run pyright/mypy
for that) — they catch the most common regressions:

- the ``py.typed`` marker was accidentally deleted or excluded from
  the wheel
- a public method we intended to annotate had its annotations stripped
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import annnet
from annnet.core._Slices import SliceManager
from annnet.core.graph import AnnNet


def test_py_typed_marker_exists() -> None:
    pkg_dir = Path(annnet.__file__).parent
    marker = pkg_dir / 'py.typed'
    assert marker.exists(), f'PEP 561 marker missing at {marker}'


def _has_annotated_signature(fn, *, must_have_return: bool = True) -> bool:
    sig = inspect.signature(fn)
    if must_have_return and sig.return_annotation is inspect.Signature.empty:
        return False
    for name, param in sig.parameters.items():
        if name == 'self' or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is inspect.Parameter.empty:
            return False
    return True


def test_annnet_core_methods_are_annotated() -> None:
    """Spot-check that the MVP set is annotated."""
    targets = [
        AnnNet.add_vertices,
        AnnNet.add_edges,
        AnnNet.remove_vertices,
        AnnNet.remove_edges,
        AnnNet.get_vertex,
        AnnNet.get_edge,
        AnnNet.has_vertex,
        AnnNet.has_edge,
        AnnNet.vertices,
        AnnNet.edges,
        AnnNet.edge_list,
        AnnNet.incident_edges,
    ]
    missing = [fn.__qualname__ for fn in targets if not _has_annotated_signature(fn)]
    assert not missing, f'unannotated AnnNet methods: {missing}'


def test_annnet_core_properties_have_return_types() -> None:
    """``G.nv``, ``G.shape``, etc. should advertise their return type."""
    property_names = [
        'nv',
        'ne',
        'num_vertices',
        'num_edges',
        'shape',
        'obs',
        'var',
        'uns',
        'slices',
        'attrs',
        'views',
        'ops',
        'layers',
        'nx',
        'ig',
        'gt',
    ]
    missing = []
    for name in property_names:
        descriptor = getattr(AnnNet, name)
        if not isinstance(descriptor, property):
            continue
        hints = typing.get_type_hints(descriptor.fget)
        if 'return' not in hints:
            missing.append(name)
    assert not missing, f'properties without return-type hints: {missing}'


def test_slice_manager_surface_is_annotated() -> None:
    """Every public method on G.slices should advertise types."""
    skipped = {'__init__', '__repr__'}
    missing = []
    for name in dir(SliceManager):
        if name.startswith('_') and name not in skipped:
            continue
        attr = getattr(SliceManager, name, None)
        if isinstance(attr, property):
            if 'return' not in typing.get_type_hints(attr.fget):
                missing.append(f'{name} (property getter)')
            continue
        if callable(attr) and not _has_annotated_signature(attr):
            missing.append(name)
    assert not missing, f'unannotated SliceManager surface: {missing}'


def test_io_and_adapter_entry_points_are_annotated() -> None:
    """Spot-check the top-level IO + adapter entry points."""
    from annnet.io.annnet_format import read as annnet_read
    from annnet.io.annnet_format import write as annnet_write
    from annnet.io.cx2 import from_cx2, to_cx2
    from annnet.io.csv_format import edges_to_csv, from_csv, from_dataframe
    from annnet.io.dataframes import from_dataframes, to_dataframes
    from annnet.io.json_format import from_json, to_json

    callables = [
        to_json,
        from_json,
        annnet_write,
        annnet_read,
        to_dataframes,
        from_dataframes,
        to_cx2,
        from_cx2,
        from_csv,
        from_dataframe,
        edges_to_csv,
    ]
    try:
        from annnet.adapters.networkx_adapter import from_nx, to_nx

        callables.extend([to_nx, from_nx])
    except ModuleNotFoundError:
        pass
    try:
        from annnet.adapters.igraph_adapter import from_igraph, to_igraph

        callables.extend([to_igraph, from_igraph])
    except ModuleNotFoundError:
        pass
    try:
        from annnet.adapters.graphtool_adapter import from_graphtool, to_graphtool

        callables.extend([to_graphtool, from_graphtool])
    except ModuleNotFoundError:
        pass

    missing = [fn.__qualname__ for fn in callables if not _has_annotated_signature(fn)]
    assert not missing, f'unannotated IO/adapter entry points: {missing}'
