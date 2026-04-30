from __future__ import annotations

from typing import Any

import narwhals as nw
from narwhals.typing import IntoDataFrame

from ..core.graph import AnnNet
from .._support.dataframe_backend import dataframe_height, dataframe_to_rows, dataframe_from_rows


def _binary_edge_rows(graph: AnnNet, edge_attrs: dict, *, public_only: bool):
    edge_weights = graph.edge_weights
    edge_directed = graph.edge_directed
    default_directed = True if graph.directed is None else graph.directed

    for eid, (src, tgt, etype) in graph.edge_definitions.items():
        row = {
            'edge_id': eid,
            'source': src,
            'target': tgt,
            'weight': 1.0 if edge_weights.get(eid) is None else float(edge_weights[eid]),
            'directed': edge_directed.get(eid, default_directed),
            'edge_type': etype,
        }

        attrs = edge_attrs.get(eid)
        if attrs:
            attr_dict = dict(attrs)
            attr_dict.pop('edge_id', None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() if not str(k).startswith('__')}
            row.update(attr_dict)

        yield row


def _hyperedge_rows(graph: AnnNet, edge_attrs: dict, *, public_only: bool, explode: bool):
    edge_weights = graph.edge_weights

    for eid, spec in graph.hyperedge_definitions.items():
        directed = bool(spec.get('directed', False))
        weight = 1.0 if edge_weights.get(eid) is None else float(edge_weights[eid])
        attrs = edge_attrs.get(eid)
        attr_dict = {}
        if attrs:
            attr_dict = dict(attrs)
            attr_dict.pop('edge_id', None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() if not str(k).startswith('__')}

        if explode:
            if directed:
                for vertex_id in spec.get('head', []):
                    row = {
                        'edge_id': eid,
                        'vertex_id': vertex_id,
                        'role': 'head',
                        'weight': weight,
                        'directed': True,
                    }
                    row.update(attr_dict)
                    yield row

                for vertex_id in spec.get('tail', []):
                    row = {
                        'edge_id': eid,
                        'vertex_id': vertex_id,
                        'role': 'tail',
                        'weight': weight,
                        'directed': True,
                    }
                    row.update(attr_dict)
                    yield row
            else:
                for vertex_id in spec.get('members', []):
                    row = {
                        'edge_id': eid,
                        'vertex_id': vertex_id,
                        'role': 'member',
                        'weight': weight,
                        'directed': False,
                    }
                    row.update(attr_dict)
                    yield row
            continue

        row = {
            'edge_id': eid,
            'directed': directed,
            'weight': weight,
            'head': list(spec.get('head', [])) if directed else None,
            'tail': list(spec.get('tail', [])) if directed else None,
            'members': None if directed else list(spec.get('members', [])),
        }
        row.update(attr_dict)
        yield row


def to_dataframes(
    graph: AnnNet,
    *,
    include_slices: bool = True,
    include_hyperedges: bool = True,
    explode_hyperedges: bool = False,
    public_only: bool = True,
) -> dict[str, Any]:
    """Export graph to DataFrames using AnnNet's selected dataframe backend.

    Returns a dictionary of DataFrames representing different aspects of the graph:
    - 'nodes': Vertex IDs and attributes
    - 'edges': Binary edges with source, target, weight, directed, attributes
    - 'hyperedges': Hyperedges with head/tail sets (if include_hyperedges=True)
    - 'slices': slice membership (if include_slices=True)
    - 'slice_weights': Per-slice edge weights (if include_slices=True)

    Args:
        graph: AnnNet instance to export
        include_slices: Include slice membership tables
        include_hyperedges: Include hyperedge table
        explode_hyperedges: If True, explode hyperedges to one row per endpoint
        public_only: If True, filter out attributes starting with '__'

    Returns
    -------
        Dictionary mapping table names to DataFrames.

    """
    result = {}
    backend = getattr(graph, '_annotations_backend', 'auto')
    vertex_attrs = {
        row.get('vertex_id'): row
        for row in dataframe_to_rows(graph.vertex_attributes)
        if row.get('vertex_id') is not None
    }
    edge_attrs = {
        row.get('edge_id'): row
        for row in dataframe_to_rows(graph.edge_attributes)
        if row.get('edge_id') is not None
    }

    # 1. Nodes table
    nodes_data = []
    for vid in graph.vertices():
        row = {'vertex_id': vid}
        attrs = vertex_attrs.get(vid)
        if attrs:
            attr_dict = dict(attrs)
            attr_dict.pop('vertex_id', None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() if not str(k).startswith('__')}
            row.update(attr_dict)
        nodes_data.append(row)

    result['nodes'] = dataframe_from_rows(
        nodes_data,
        schema={'vertex_id': 'text'},
        backend=backend,
    )

    # 2. Binary edges table
    edges_data = list(_binary_edge_rows(graph, edge_attrs, public_only=public_only))

    result['edges'] = dataframe_from_rows(
        edges_data,
        schema={
            'edge_id': 'text',
            'source': 'text',
            'target': 'text',
            'weight': 'float',
            'directed': 'bool',
            'edge_type': 'text',
        },
        backend=backend,
    )

    # 3. Hyperedges table
    if include_hyperedges:
        hyperedges_data = list(
            _hyperedge_rows(
                graph,
                edge_attrs,
                public_only=public_only,
                explode=explode_hyperedges,
            )
        )

        if explode_hyperedges:
            result['hyperedges'] = dataframe_from_rows(
                hyperedges_data,
                schema={
                    'edge_id': 'text',
                    'vertex_id': 'text',
                    'role': 'text',
                    'weight': 'float',
                    'directed': 'bool',
                },
                backend=backend,
            )
        else:
            result['hyperedges'] = dataframe_from_rows(
                hyperedges_data,
                schema={
                    'edge_id': 'text',
                    'directed': 'bool',
                    'weight': 'float',
                    'head': 'list_text',
                    'tail': 'list_text',
                    'members': 'list_text',
                },
                backend=backend,
            )

    # 4. Slice membership
    if include_slices:
        slices_data = []
        for lid in graph.slices.list_slices(include_default=True):
            for eid in graph.slices.edges(lid):
                slices_data.append({'slice_id': lid, 'edge_id': eid})

        result['slices'] = dataframe_from_rows(
            slices_data,
            schema={'slice_id': 'text', 'edge_id': 'text'},
            backend=backend,
        )

        # 5. Per-slice weights
        slice_weights_data = []
        for row in dataframe_to_rows(getattr(graph, 'edge_slice_attributes', None)):
            if {'slice_id', 'edge_id', 'weight'}.issubset(row):
                slice_weights_data.append(
                    {
                        'slice_id': row['slice_id'],
                        'edge_id': row['edge_id'],
                        'weight': row['weight'],
                    }
                )

        result['slice_weights'] = dataframe_from_rows(
            slice_weights_data,
            schema={'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
            backend=backend,
        )

    return result


def _to_dicts(df: nw.DataFrame[Any]) -> list[dict[str, Any]]:
    """Convert narwhals DataFrame to list of dicts."""
    return dataframe_to_rows(nw.to_native(df))


def _get_height(df: nw.DataFrame[Any]) -> int:
    """Get row count from narwhals DataFrame."""
    return dataframe_height(nw.to_native(df))


def from_dataframes(
    nodes: IntoDataFrame | None = None,
    edges: IntoDataFrame | None = None,
    hyperedges: IntoDataFrame | None = None,
    slices: IntoDataFrame | None = None,
    slice_weights: IntoDataFrame | None = None,
    *,
    directed: bool | None = None,
    exploded_hyperedges: bool = False,
) -> AnnNet:
    """Import graph from any DataFrame (Pandas, Polars, PyArrow, etc.).

    Accepts DataFrames in the format produced by to_dataframes():

    Nodes DataFrame (optional):
        - Required: vertex_id
        - Optional: any attribute columns

    Edges DataFrame (optional):
        - Required: source, target
        - Optional: edge_id, weight, directed, edge_type, attribute columns

    Hyperedges DataFrame (optional):
        - Compact format: edge_id, directed, weight, head (list), tail (list), members (list)
        - Exploded format: edge_id, vertex_id, role, weight, directed

    slices DataFrame (optional):
        - Required: slice_id, edge_id

    slice_weights DataFrame (optional):
        - Required: slice_id, edge_id, weight

    Args:
        nodes: DataFrame with vertex_id and attributes (Pandas/Polars/PyArrow/etc.)
        edges: DataFrame with binary edges
        hyperedges: DataFrame with hyperedges
        slices: DataFrame with slice membership
        slice_weights: DataFrame with per-slice edge weights
        directed: Default directedness (None = mixed graph)
        exploded_hyperedges: If True, hyperedges DataFrame is in exploded format

    Returns
    -------
        AnnNet instance

    """
    G = AnnNet(directed=directed)

    # 1. Add vertices
    if nodes is not None:
        nodes_nw = nw.from_native(nodes, eager_only=True)
        if _get_height(nodes_nw) > 0:
            if 'vertex_id' not in nodes_nw.columns:
                raise ValueError("nodes DataFrame must have 'vertex_id' column")

            G.add_vertices_bulk(_to_dicts(nodes_nw))

    # 2. Add binary edges
    if edges is not None:
        edges_nw = nw.from_native(edges, eager_only=True)
        if _get_height(edges_nw) > 0:
            if 'source' not in edges_nw.columns or 'target' not in edges_nw.columns:
                raise ValueError("edges DataFrame must have 'source' and 'target' columns")

            edge_rows = []
            for row in _to_dicts(edges_nw):
                src = row.pop('source')
                tgt = row.pop('target')
                eid = row.pop('edge_id', None)
                weight = row.pop('weight', 1.0)
                edge_directed = row.pop('directed', directed)
                etype = row.pop('edge_type', 'regular')

                edge_rows.append(
                    {
                        'source': src,
                        'target': tgt,
                        'edge_id': eid,
                        'weight': weight,
                        'edge_directed': edge_directed,
                        'edge_type': etype,
                        'attributes': row,
                    }
                )

            G.add_edges_bulk(edge_rows)

    # 3. Add hyperedges
    if hyperedges is not None:
        hyperedges_nw = nw.from_native(hyperedges, eager_only=True)
        if _get_height(hyperedges_nw) > 0:
            if exploded_hyperedges:
                if (
                    'edge_id' not in hyperedges_nw.columns
                    or 'vertex_id' not in hyperedges_nw.columns
                ):
                    raise ValueError(
                        "Exploded hyperedges must have 'edge_id' and 'vertex_id' columns"
                    )

                # Group by edge_id - need to collect all rows first
                grouped: dict[str, dict[str, list[Any]]] = {}
                for row in _to_dicts(hyperedges_nw):
                    eid = row['edge_id']
                    if eid not in grouped:
                        grouped[eid] = {'vertices': [], 'roles': [], 'directed': [], 'weights': []}
                    grouped[eid]['vertices'].append(row['vertex_id'])
                    grouped[eid]['roles'].append(row.get('role', 'member'))
                    grouped[eid]['directed'].append(row.get('directed', False))
                    grouped[eid]['weights'].append(row.get('weight', 1.0))

                for eid, data in grouped.items():
                    is_directed = data['directed'][0] if data['directed'] else False
                    weight = data['weights'][0] if data['weights'] else 1.0

                    if is_directed:
                        head = [
                            v
                            for v, r in zip(data['vertices'], data['roles'], strict=False)
                            if r == 'head'
                        ]
                        tail = [
                            v
                            for v, r in zip(data['vertices'], data['roles'], strict=False)
                            if r == 'tail'
                        ]
                        G.add_edges(src=head, tgt=tail, edge_id=eid, directed=True, weight=weight)
                    else:
                        G.add_edges(
                            src=data['vertices'],
                            edge_id=eid,
                            directed=False,
                            weight=weight,
                        )
            else:
                if 'edge_id' not in hyperedges_nw.columns:
                    raise ValueError("hyperedges DataFrame must have 'edge_id' column")

                for row in _to_dicts(hyperedges_nw):
                    eid = row.pop('edge_id')
                    directed_he = row.pop('directed', False)
                    weight = row.pop('weight', 1.0)
                    head = row.pop('head', None)
                    tail = row.pop('tail', None)
                    members = row.pop('members', None)

                    if directed_he:
                        G.add_edges(
                            src=head or [],
                            tgt=tail or [],
                            edge_id=eid,
                            directed=True,
                            weight=weight,
                        )
                    else:
                        G.add_edges(
                            src=members or [],
                            edge_id=eid,
                            directed=False,
                            weight=weight,
                        )

                    if row:
                        G.attrs.set_edge_attrs(eid, **row)

    # 4. Add slice memberships
    if slices is not None:
        slices_nw = nw.from_native(slices, eager_only=True)
        if _get_height(slices_nw) > 0:
            if 'slice_id' not in slices_nw.columns or 'edge_id' not in slices_nw.columns:
                raise ValueError("slices DataFrame must have 'slice_id' and 'edge_id' columns")

            existing_slices = set(G.slices.list_slices(include_default=True))
            existing_edges = set(G.edge_definitions) | set(G.hyperedge_definitions)
            for row in _to_dicts(slices_nw):
                lid = row['slice_id']
                eid = row['edge_id']

                if lid not in existing_slices:
                    G.slices.add_slice(lid)
                    existing_slices.add(lid)

                if eid in existing_edges:
                    G.slices.add_edge_to_slice(lid, eid)

    # 5. Add per-slice weights
    if slice_weights is not None:
        slice_weights_nw = nw.from_native(slice_weights, eager_only=True)
        if _get_height(slice_weights_nw) > 0:
            cols = set(slice_weights_nw.columns)
            if {'slice_id', 'edge_id', 'weight'}.issubset(cols):
                existing_slices = set(G.slices.list_slices(include_default=True))
                existing_edges = set(G.edge_definitions) | set(G.hyperedge_definitions)
                for row in _to_dicts(slice_weights_nw):
                    lid = row['slice_id']
                    eid = row['edge_id']
                    weight = row['weight']

                    if lid in existing_slices and eid in existing_edges:
                        G.attrs.set_edge_slice_attrs(lid, eid, weight=weight)

    return G
