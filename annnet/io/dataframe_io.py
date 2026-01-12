from __future__ import annotations

from typing import Any

import narwhals as nw

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None
from narwhals.typing import IntoDataFrame

if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

try:
    from annnet.core.graph import AnnNet
except ImportError:
    from ..core.graph import AnnNet


def to_dataframes(
    graph: AnnNet,
    *,
    include_slices: bool = True,
    include_hyperedges: bool = True,
    explode_hyperedges: bool = False,
    public_only: bool = True,
) -> dict[str, pl.DataFrame]:
    """Export graph to Polars DataFrames.

    Returns a dictionary of DataFrames representing different aspects of the graph:
    - 'nodes': Vertex IDs and attributes
    - 'edges': Binary edges with source, target, weight, directed, attributes
    - 'hyperedges': Hyperedges with head/tail sets (if include_hyperedges=True)
    - 'slices': slice membership (if include_slices=True)
    - 'slice_weights': Per-slice edge weights (if include_slices=True)

    Note: Output is always Polars because hyperedges use List types which
    aren't universally supported across dataframe libraries.

    Args:
        graph: AnnNet instance to export
        include_slices: Include slice membership tables
        include_hyperedges: Include hyperedge table
        explode_hyperedges: If True, explode hyperedges to one row per endpoint
        public_only: If True, filter out attributes starting with '__'

    Returns:
        Dictionary mapping table names to Polars DataFrames

    """
    result = {}

    # 1. Nodes table
    nodes_data = []
    for vid in graph.vertices():
        row = {"vertex_id": vid}
        attrs = graph.vertex_attributes.filter(pl.col("vertex_id") == vid).to_dicts()
        if attrs:
            attr_dict = dict(attrs[0])
            attr_dict.pop("vertex_id", None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() if not str(k).startswith("__")}
            row.update(attr_dict)
        nodes_data.append(row)

    result["nodes"] = (
        pl.DataFrame(nodes_data) if nodes_data else pl.DataFrame(schema={"vertex_id": pl.Utf8})
    )

    # 2. Binary edges table
    edges_data = []
    for eid, (src, tgt, etype) in graph.edge_definitions.items():
        if etype == "hyper":
            continue

        row = {
            "edge_id": eid,
            "source": src,
            "target": tgt,
            "weight": graph.edge_weights.get(eid, 1.0),
            "directed": graph.edge_directed.get(
                eid, True if graph.directed is None else graph.directed
            ),
            "edge_type": etype,
        }

        attrs = graph.edge_attributes.filter(pl.col("edge_id") == eid).to_dicts()
        if attrs:
            attr_dict = dict(attrs[0])
            attr_dict.pop("edge_id", None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() if not str(k).startswith("__")}
            row.update(attr_dict)

        edges_data.append(row)

    result["edges"] = (
        pl.DataFrame(edges_data)
        if edges_data
        else pl.DataFrame(
            schema={
                "edge_id": pl.Utf8,
                "source": pl.Utf8,
                "target": pl.Utf8,
                "weight": pl.Float64,
                "directed": pl.Boolean,
                "edge_type": pl.Utf8,
            }
        )
    )

    # 3. Hyperedges table
    if include_hyperedges:
        hyperedges_data = []

        if explode_hyperedges:
            for eid, meta in graph.hyperedge_definitions.items():
                directed = meta.get("directed", False)
                weight = graph.edge_weights.get(eid, 1.0)

                attrs = graph.edge_attributes.filter(pl.col("edge_id") == eid).to_dicts()
                attr_dict = {}
                if attrs:
                    attr_dict = dict(attrs[0])
                    attr_dict.pop("edge_id", None)
                    if public_only:
                        attr_dict = {
                            k: v for k, v in attr_dict.items() if not str(k).startswith("__")
                        }

                if directed:
                    for v in meta.get("head", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "head",
                            "weight": weight,
                            "directed": True,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)

                    for v in meta.get("tail", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "tail",
                            "weight": weight,
                            "directed": True,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)
                else:
                    for v in meta.get("members", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "member",
                            "weight": weight,
                            "directed": False,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)
        else:
            for eid, meta in graph.hyperedge_definitions.items():
                directed = meta.get("directed", False)
                weight = graph.edge_weights.get(eid, 1.0)

                row = {
                    "edge_id": eid,
                    "directed": directed,
                    "weight": weight,
                }

                if directed:
                    row["head"] = list(meta.get("head", []))
                    row["tail"] = list(meta.get("tail", []))
                    row["members"] = None
                else:
                    row["head"] = None
                    row["tail"] = None
                    row["members"] = list(meta.get("members", []))

                attrs = graph.edge_attributes.filter(pl.col("edge_id") == eid).to_dicts()
                if attrs:
                    attr_dict = dict(attrs[0])
                    attr_dict.pop("edge_id", None)
                    if public_only:
                        attr_dict = {
                            k: v for k, v in attr_dict.items() if not str(k).startswith("__")
                        }
                    row.update(attr_dict)

                hyperedges_data.append(row)

        if hyperedges_data:
            result["hyperedges"] = pl.DataFrame(hyperedges_data)
        else:
            if explode_hyperedges:
                result["hyperedges"] = pl.DataFrame(
                    schema={
                        "edge_id": pl.Utf8,
                        "vertex_id": pl.Utf8,
                        "role": pl.Utf8,
                        "weight": pl.Float64,
                        "directed": pl.Boolean,
                    }
                )
            else:
                result["hyperedges"] = pl.DataFrame(
                    schema={
                        "edge_id": pl.Utf8,
                        "directed": pl.Boolean,
                        "weight": pl.Float64,
                        "head": pl.List(pl.Utf8),
                        "tail": pl.List(pl.Utf8),
                        "members": pl.List(pl.Utf8),
                    }
                )

    # 4. Slice membership
    if include_slices:
        slices_data = []
        try:
            for lid in graph.list_slices(include_default=True):
                slice_meta = graph._slices.get(lid, {})
                for eid in slice_meta.get("edges", []):
                    slices_data.append({"slice_id": lid, "edge_id": eid})
        except Exception:
            pass

        result["slices"] = (
            pl.DataFrame(slices_data)
            if slices_data
            else pl.DataFrame(schema={"slice_id": pl.Utf8, "edge_id": pl.Utf8})
        )

        # 5. Per-slice weights
        slice_weights_data = []
        try:
            df = graph.edge_slice_attributes
            if isinstance(df, pl.DataFrame) and df.height > 0:
                if {"slice_id", "edge_id", "weight"}.issubset(df.columns):
                    slice_weights_data = df.select(["slice_id", "edge_id", "weight"]).to_dicts()
        except Exception:
            pass

        result["slice_weights"] = (
            pl.DataFrame(slice_weights_data)
            if slice_weights_data
            else pl.DataFrame(
                schema={"slice_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64}
            )
        )

    return result


def _to_dicts(df: nw.DataFrame[Any]) -> list[dict[str, Any]]:
    """Convert narwhals DataFrame to list of dicts."""
    return [dict(zip(df.columns, row)) for row in df.rows()]


def _get_height(df: nw.DataFrame[Any]) -> int:
    """Get row count from narwhals DataFrame."""
    return df.shape[0]


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

    Returns:
        AnnNet instance

    """
    G = AnnNet(directed=directed)

    # 1. Add vertices
    if nodes is not None:
        nodes_nw = nw.from_native(nodes, eager_only=True)
        if _get_height(nodes_nw) > 0:
            if "vertex_id" not in nodes_nw.columns:
                raise ValueError("nodes DataFrame must have 'vertex_id' column")

            for row in _to_dicts(nodes_nw):
                vid = row.pop("vertex_id")
                G.add_vertex(vid)
                if row:
                    G.set_vertex_attrs(vid, **row)

    # 2. Add binary edges
    if edges is not None:
        edges_nw = nw.from_native(edges, eager_only=True)
        if _get_height(edges_nw) > 0:
            if "source" not in edges_nw.columns or "target" not in edges_nw.columns:
                raise ValueError("edges DataFrame must have 'source' and 'target' columns")

            edge_rows = []
            for row in _to_dicts(edges_nw):
                src = row.pop("source")
                tgt = row.pop("target")
                eid = row.pop("edge_id", None)
                weight = row.pop("weight", 1.0)
                edge_directed = row.pop("directed", directed)
                etype = row.pop("edge_type", "regular")

                edge_rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "edge_id": eid,
                        "weight": weight,
                        "edge_directed": edge_directed,
                        "edge_type": etype,
                        "attributes": row,
                    }
                )

            G.add_edges_bulk(edge_rows)

    # 3. Add hyperedges
    if hyperedges is not None:
        hyperedges_nw = nw.from_native(hyperedges, eager_only=True)
        if _get_height(hyperedges_nw) > 0:
            if exploded_hyperedges:
                if (
                    "edge_id" not in hyperedges_nw.columns
                    or "vertex_id" not in hyperedges_nw.columns
                ):
                    raise ValueError(
                        "Exploded hyperedges must have 'edge_id' and 'vertex_id' columns"
                    )

                # Group by edge_id - need to collect all rows first
                grouped: dict[str, dict[str, list[Any]]] = {}
                for row in _to_dicts(hyperedges_nw):
                    eid = row["edge_id"]
                    if eid not in grouped:
                        grouped[eid] = {"vertices": [], "roles": [], "directed": [], "weights": []}
                    grouped[eid]["vertices"].append(row["vertex_id"])
                    grouped[eid]["roles"].append(row.get("role", "member"))
                    grouped[eid]["directed"].append(row.get("directed", False))
                    grouped[eid]["weights"].append(row.get("weight", 1.0))

                for eid, data in grouped.items():
                    is_directed = data["directed"][0] if data["directed"] else False
                    weight = data["weights"][0] if data["weights"] else 1.0

                    if is_directed:
                        head = [v for v, r in zip(data["vertices"], data["roles"]) if r == "head"]
                        tail = [v for v, r in zip(data["vertices"], data["roles"]) if r == "tail"]
                        G.add_hyperedge(
                            head=head, tail=tail, edge_id=eid, edge_directed=True, weight=weight
                        )
                    else:
                        G.add_hyperedge(
                            members=data["vertices"],
                            edge_id=eid,
                            edge_directed=False,
                            weight=weight,
                        )
            else:
                if "edge_id" not in hyperedges_nw.columns:
                    raise ValueError("hyperedges DataFrame must have 'edge_id' column")

                for row in _to_dicts(hyperedges_nw):
                    eid = row.pop("edge_id")
                    directed_he = row.pop("directed", False)
                    weight = row.pop("weight", 1.0)
                    head = row.pop("head", None)
                    tail = row.pop("tail", None)
                    members = row.pop("members", None)

                    if directed_he:
                        G.add_hyperedge(
                            head=head or [],
                            tail=tail or [],
                            edge_id=eid,
                            edge_directed=True,
                            weight=weight,
                        )
                    else:
                        G.add_hyperedge(
                            members=members or [],
                            edge_id=eid,
                            edge_directed=False,
                            weight=weight,
                        )

                    if row:
                        G.set_edge_attrs(eid, **row)

    # 4. Add slice memberships
    if slices is not None:
        slices_nw = nw.from_native(slices, eager_only=True)
        if _get_height(slices_nw) > 0:
            if "slice_id" not in slices_nw.columns or "edge_id" not in slices_nw.columns:
                raise ValueError("slices DataFrame must have 'slice_id' and 'edge_id' columns")

            for row in _to_dicts(slices_nw):
                lid = row["slice_id"]
                eid = row["edge_id"]

                try:
                    if lid not in set(G.list_slices(include_default=True)):
                        G.add_slice(lid)
                except Exception:
                    G.add_slice(lid)

                try:
                    G.add_edge_to_slice(lid, eid)
                except Exception:
                    pass

    # 5. Add per-slice weights
    if slice_weights is not None:
        slice_weights_nw = nw.from_native(slice_weights, eager_only=True)
        if _get_height(slice_weights_nw) > 0:
            cols = set(slice_weights_nw.columns)
            if {"slice_id", "edge_id", "weight"}.issubset(cols):
                for row in _to_dicts(slice_weights_nw):
                    lid = row["slice_id"]
                    eid = row["edge_id"]
                    weight = row["weight"]

                    try:
                        G.set_edge_slice_attrs(lid, eid, weight=weight)
                    except Exception:
                        pass

    return G
