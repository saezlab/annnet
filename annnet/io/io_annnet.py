from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy as scipy
import scipy.sparse as sp

from ._utils import _read_archive, _write_archive

if TYPE_CHECKING:
    from ..core.graph import AnnNet

try:
    _write_cache  # type: ignore[name-defined]
except NameError:

    def _write_cache(*args, **kwargs):  # type: ignore[no-redef]
        raise NotImplementedError(
            "_write_cache() was referenced but is not defined. "
            "Replace this call with the actual writer function for AnnNet IO."
        )


def _have_polars():
    try:
        import polars as pl  # noqa

        return True
    except Exception:
        return False


def _df_from_dict(data: dict):
    """
    Create a DataFrame from dict-of-columns using:
    - polars if available
    - else pandas
    """
    if _have_polars():
        import polars as pl

        return pl.DataFrame(data)
    import pandas as pd

    return pd.DataFrame(data)


def _write_parquet_df(df, path, *, compression="zstd"):
    """
    Write a dataframe-like object to parquet:
    - polars DataFrame: df.write_parquet
    - pandas DataFrame: df.to_parquet(engine="pyarrow")
    - anything else: try narwhals -> pandas
    """
    # Polars fast-path
    if hasattr(df, "write_parquet"):
        return df.write_parquet(path, compression=compression)

    # Pandas path
    if hasattr(df, "to_parquet"):
        try:
            return df.to_parquet(path, engine="pyarrow", compression=compression, index=False)
        except TypeError:
            # older pandas/pyarrow combos: compression may differ
            return df.to_parquet(path, engine="pyarrow", index=False)

    # Last resort: narwhals -> pandas
    import narwhals as nw

    native = nw.to_native(nw.from_native(df, pass_through=True))
    if hasattr(native, "to_parquet"):
        try:
            return native.to_parquet(path, engine="pyarrow", compression=compression, index=False)
        except TypeError:
            return native.to_parquet(path, engine="pyarrow", index=False)

    raise TypeError(f"Don't know how to write parquet for type: {type(df)!r}")


def _read_parquet(path):
    """
    Read parquet into:
    - polars DataFrame if available
    - else pandas DataFrame
    """
    if _have_polars():
        import polars as pl

        return pl.read_parquet(path)
    import pandas as pd

    return pd.read_parquet(path, engine="pyarrow")


def _iter_rows(df):
    """
    Return rows as list[dict] for both:
      - polars DataFrame: .to_dicts()
      - pandas DataFrame: .to_dict(orient="records")
    """
    if df is None:
        return []
    # Polars
    if hasattr(df, "to_dicts") and callable(getattr(df, "to_dicts")):
        return df.to_dicts()
    # Pandas
    if hasattr(df, "to_dict") and callable(getattr(df, "to_dict")):
        try:
            return df.to_dict(orient="records")
        except TypeError:
            # very old pandas fallback
            return [dict(zip(df.columns, r)) for r in df.itertuples(index=False, name=None)]
    # Last resort: try narwhals -> pandas/polars native
    try:
        import narwhals as nw

        native = nw.to_native(nw.from_native(df, pass_through=True))
        if hasattr(native, "to_dicts") and callable(getattr(native, "to_dicts")):
            return native.to_dicts()
        if hasattr(native, "to_dict") and callable(getattr(native, "to_dict")):
            return native.to_dict(orient="records")
    except Exception:
        pass
    raise TypeError(f"Unsupported dataframe type for row iteration: {type(df)!r}")


ANNNET_EXT = "graph.annnet"


def _write_dir(graph, path: str | Path, *, compression="zstd", overwrite=False):
    """Write graph to disk with zero topology loss.

    Parameters
    ----------
    path : str | Path
        Target directory (e.g., "my_graph.annnet")
    compression : str, default "zstd"
        Compression codec for Zarr/Parquet
    overwrite : bool, default False
        Allow overwriting existing directory

    Notes
    -----
    Creates a self-contained directory with:
    - Zarr arrays for sparse matrices
    - Parquet tables for attributes/metadata
    - JSON for unstructured data

    """
    import json
    from pathlib import Path

    root = Path(path)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Set overwrite=True.")
    else:
        root.mkdir(parents=True, exist_ok=True)

    # 1. Write manifest
    manifest = {
        "format": "annnet",
        "version": "1.0.0",
        "created": datetime.now(UTC).isoformat(),
        "annnet_version": "0.1.0",
        "graph_version": graph._version,
        "directed": graph.directed,
        "counts": {
            "vertices": sum(1 for t in graph.entity_types.values() if t == "vertex"),
            "edges": graph._num_edges,
            "entities": graph._num_entities,
            "slices": len(graph._slices),
            "hyperedges": sum(1 for k in graph.edge_kind.values() if k == "hyper"),
            "aspects": len(getattr(graph, "aspects", [])),
        },
        "slices": list(graph._slices.keys()),
        "active_slice": graph._current_slice,
        "default_slice": graph._default_slice,
        "compression": compression,
        # make encoding explicit for tests/docs
        "encoding": {"zarr": "v3", "parquet": "2.0"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 2. Write structure/ (topology)
    _write_structure(graph, root / "structure", compression)

    # 3. Write tables/ (Polars > Parquet)
    _write_tables(graph, root / "tables", compression)

    # 4. Write layers/ (Kivela Multilayer structures)
    _write_multilayers(graph, root / "layers", compression)

    # 5. Write slices/
    _write_slices(graph, root / "slices", compression)

    # 6. Write audit/
    _write_audit(graph, root / "audit", compression)

    # 7. Write uns/
    _write_uns(graph, root / "uns")

    # 8. Optional: Write cache/
    if hasattr(graph, "_cached_csr") or hasattr(graph, "_cached_csc"):
        _write_cache(graph, root / "cache", compression)


def write(graph, path: str | Path, *, compression="zstd", overwrite=False):
    path = Path(path)

    # FILE MODE (.annnet archive)
    if path.suffix == ".annnet" and not path.is_dir():
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Set overwrite=True.")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp) / "graph.annnet"
            _write_dir(graph, tmp_root, compression=compression, overwrite=True)
            _write_archive(tmp_root, path)
        return

    # DIRECTORY MODE (canonical format)
    return _write_dir(graph, path, compression=compression, overwrite=overwrite)


def _write_structure(graph, path: Path, compression: str):
    """Write sparse incidence matrix + all index mappings."""
    import zarr

    path.mkdir(parents=True, exist_ok=True)

    # Convert DOK > COO for efficient storage
    coo = graph._matrix.tocoo()

    # Write incidence matrix as Zarr (chunked, compressed)
    inc_path = path / "incidence.zarr"

    # Zarr v3 compatibility
    import numpy as np

    root = zarr.open_group(str(inc_path), mode="w")

    from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

    codec = BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)

    row = np.asarray(coo.row, dtype=np.int32)
    col = np.asarray(coo.col, dtype=np.int32)
    dat = np.asarray(coo.data, dtype=np.float32)

    root.create_array("row", data=row, chunks=(10000,), compressors=[codec])
    root.create_array("col", data=col, chunks=(10000,), compressors=[codec])
    root.create_array("data", data=dat, chunks=(10000,), compressors=[codec])
    root.attrs["shape"] = coo.shape

    # Write all index mappings as Parquet
    def dict_to_parquet(d: dict, filepath: Path, id_name: str, val_name: str):
        try:
            import polars as pl  # optional
        except Exception:  # ModuleNotFoundError, etc.
            pl = None
        df = _df_from_dict({id_name: list(d.keys()), val_name: list(d.values())})
        _write_parquet_df(df, filepath, compression=compression)

    dict_to_parquet(graph.entity_to_idx, path / "entity_to_idx.parquet", "entity_id", "idx")
    dict_to_parquet(graph.idx_to_entity, path / "idx_to_entity.parquet", "idx", "entity_id")
    dict_to_parquet(graph.entity_types, path / "entity_types.parquet", "entity_id", "type")
    dict_to_parquet(graph.edge_to_idx, path / "edge_to_idx.parquet", "edge_id", "idx")
    dict_to_parquet(graph.idx_to_edge, path / "idx_to_edge.parquet", "idx", "edge_id")
    dict_to_parquet(graph.edge_weights, path / "edge_weights.parquet", "edge_id", "weight")
    dict_to_parquet(graph.edge_directed, path / "edge_directed.parquet", "edge_id", "directed")
    dict_to_parquet(graph.edge_kind, path / "edge_kind.parquet", "edge_id", "kind")

    # Edge definitions (tuples > struct column)

    edge_def_df = _df_from_dict(
        {
            "edge_id": list(graph.edge_definitions.keys()),
            "source": [v[0] for v in graph.edge_definitions.values()],
            "target": [v[1] for v in graph.edge_definitions.values()],
            "edge_type": [v[2] for v in graph.edge_definitions.values()],
        }
    )
    _write_parquet_df(edge_def_df, path / "edge_definitions.parquet", compression=compression)

    # Hyperedge definitions (lists > list column)
    if graph.hyperedge_definitions:
        eids, dirs, mems, heads, tails = [], [], [], [], []

        for eid, h in graph.hyperedge_definitions.items():
            eids.append(eid)
            is_dir = bool(h.get("directed", False))
            dirs.append(is_dir)

            if is_dir:
                # directed hyperedge: store head/tail lists; no members
                heads.append(sorted(map(str, h.get("head", ()))))
                tails.append(sorted(map(str, h.get("tail", ()))))
                mems.append(None)
            else:
                # undirected hyperedge: store members list; no head/tail
                heads.append(None)
                tails.append(None)
                mems.append(sorted(map(str, h.get("members", ()))))

        # Polars: keep explicit List(Utf8) dtypes; Non-polars: let pandas/pyarrow infer list columns.
        if _have_polars():
            import polars as pl

            hyper_df = pl.DataFrame({"edge_id": eids, "directed": dirs}).with_columns(
                pl.Series("members", mems, dtype=pl.List(pl.Utf8)),
                pl.Series("head", heads, dtype=pl.List(pl.Utf8)),
                pl.Series("tail", tails, dtype=pl.List(pl.Utf8)),
            )
            hyper_df.write_parquet(path / "hyperedge_definitions.parquet", compression=compression)
        else:
            hyper_df = _df_from_dict(
                {
                    "edge_id": eids,
                    "directed": dirs,
                    "members": mems,
                    "head": heads,
                    "tail": tails,
                }
            )
            _write_parquet_df(
                hyper_df, path / "hyperedge_definitions.parquet", compression=compression
            )


def _write_tables(graph, path: Path, compression: str):
    path.mkdir(parents=True, exist_ok=True)

    _write_parquet_df(
        graph.vertex_attributes, path / "vertex_attributes.parquet", compression=compression
    )
    _write_parquet_df(
        graph.edge_attributes, path / "edge_attributes.parquet", compression=compression
    )
    _write_parquet_df(
        graph.slice_attributes, path / "slice_attributes.parquet", compression=compression
    )
    _write_parquet_df(
        graph.edge_slice_attributes, path / "edge_slice_attributes.parquet", compression=compression
    )


def _write_multilayers(graph, path: Path, compression: str):
    """Write Kivela multilayer structures to disk."""
    import json

    # If no aspects are defined, skip creating the folder
    if not getattr(graph, "aspects", []):
        return

    path.mkdir(parents=True, exist_ok=True)

    # 1. Metadata: Aspects & Elementary definitions
    metadata = {
        "aspects": graph.aspects,
        "elem_layers": graph.elem_layers,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # 2. Vertex Presence (V_M)
    # Store "layer" as list[str] (works in polars; works in pandas+pyarrow as list column)
    vm_data = [{"vertex_id": u, "layer": list(aa)} for u, aa in graph._VM]

    if _have_polars():
        import polars as pl

        if not vm_data:
            vm_df = pl.DataFrame(
                {
                    "vertex_id": pl.Series([], dtype=pl.Utf8),
                    "layer": pl.Series([], dtype=pl.List(pl.Utf8)),
                }
            )
        else:
            vm_df = pl.DataFrame(vm_data)
        vm_df.write_parquet(path / "vertex_presence.parquet", compression=compression)
    else:
        # pandas fallback: keep schema stable even when empty
        if not vm_data:
            vm_df = _df_from_dict({"vertex_id": [], "layer": []})
        else:
            vm_df = _df_from_dict(
                {
                    "vertex_id": [r["vertex_id"] for r in vm_data],
                    "layer": [r["layer"] for r in vm_data],
                }
            )
        _write_parquet_df(vm_df, path / "vertex_presence.parquet", compression=compression)

    # 3. Edge Layers
    # Logic: Intra edges have 1 layer tuple. Inter/Coupling have 2.
    # We flatten this to columns: edge_id, layer_1, layer_2 (nullable)
    eids, l1s, l2s = [], [], []
    for eid, layers in graph.edge_layers.items():
        eids.append(eid)
        if isinstance(layers, tuple) and layers and isinstance(layers[0], tuple):
            # Case: Inter/Coupling -> ((a,b), (c,d))
            l1s.append(list(layers[0]))
            l2s.append(list(layers[1]))
        else:
            # Case: Intra -> (a,b)
            l1s.append(list(layers))
            l2s.append(None)

    el_df = _df_from_dict({"edge_id": eids, "layer_1": l1s, "layer_2": l2s})
    _write_parquet_df(el_df, path / "edge_layers.parquet", compression=compression)

    # 4. Attributes (Specific Layer Stores)

    # 4a. Elementary Layer Attributes (Already a Polars DF in graph.py)
    if hasattr(graph, "layer_attributes") and graph.layer_attributes is not None:
        # polars has is_empty(); pandas doesn't. handle both.
        try:
            empty = bool(getattr(graph.layer_attributes, "is_empty")())
        except Exception:
            try:
                empty = len(graph.layer_attributes) == 0
            except Exception:
                empty = False
        if not empty:
            _write_parquet_df(
                graph.layer_attributes,
                path / "elem_layer_attributes.parquet",
                compression=compression,
            )

    # 4b. Aspect Attributes (Dict -> JSON)
    if hasattr(graph, "_aspect_attrs") and graph._aspect_attrs:
        (path / "aspect_attributes.json").write_text(json.dumps(graph._aspect_attrs, indent=2))

    # 4c. Tuple Layer Attributes (Dict -> Parquet due to complex keys)
    if hasattr(graph, "_layer_attrs") and graph._layer_attrs:
        la_data = [
            {"layer": list(aa), "attributes": json.dumps(attrs)}
            for aa, attrs in graph._layer_attrs.items()
        ]
        la_df = _df_from_dict(la_data)
        _write_parquet_df(la_df, path / "tuple_layer_attributes.parquet", compression=compression)

    # 4d. Vertex-Layer Attributes
    if hasattr(graph, "_vertex_layer_attrs") and graph._vertex_layer_attrs:
        vla_data = [
            {"vertex_id": u, "layer": list(aa), "attributes": json.dumps(attrs)}
            for (u, aa), attrs in graph._vertex_layer_attrs.items()
        ]
        vla_df = _df_from_dict(vla_data)
        _write_parquet_df(vla_df, path / "vertex_layer_attributes.parquet", compression=compression)


def _write_slices(graph, path: Path, compression: str):
    """Write slice registry and memberships."""
    import json

    path.mkdir(parents=True, exist_ok=True)

    # Registry: slice_id > attributes
    registry_data = []
    for slice_id, slice_data in graph._slices.items():
        registry_data.append(
            {"slice_id": slice_id, "attributes": json.dumps(slice_data.get("attributes", {}))}
        )
    reg_df = _df_from_dict(registry_data)
    _write_parquet_df(reg_df, path / "registry.parquet", compression=compression)

    # Vertex memberships: long format
    vertex_members = []
    for slice_id, slice_data in graph._slices.items():
        for vertex_id in slice_data["vertices"]:
            vertex_members.append({"slice_id": slice_id, "vertex_id": vertex_id})
    vm_df = _df_from_dict(vertex_members)
    _write_parquet_df(vm_df, path / "vertex_memberships.parquet", compression=compression)

    # Edge memberships with weights
    edge_members: list[dict] = []
    # Primary: explicit per-slice weights (if present)
    for slice_id, edge_weights in getattr(graph, "slice_edge_weights", {}).items():
        for edge_id, weight in edge_weights.items():
            edge_members.append({"slice_id": slice_id, "edge_id": edge_id, "weight": weight})
    # Fallback: derive from registered slice edges if no explicit weights
    if not edge_members:
        for slice_id, slice_data in graph._slices.items():
            for edge_id in slice_data.get("edges", ()):
                edge_members.append({"slice_id": slice_id, "edge_id": edge_id, "weight": None})
    # Ensure a stable schema even if there are zero rows
    if edge_members:
        em_df = _df_from_dict(edge_members)
    else:
        # explicit empty schema for pandas too
        em_df = _df_from_dict({"slice_id": [], "edge_id": [], "weight": []})
    _write_parquet_df(em_df, path / "edge_memberships.parquet", compression=compression)


def _write_audit(graph, path: Path, compression: str):
    """Write history, snapshots, provenance."""
    import json
    import sys
    from datetime import datetime

    try:
        from datetime import UTC  # Py3.11+
    except Exception:  # fallback for <3.11
        from datetime import timezone as _tz

        UTC = UTC

    import numpy as np

    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None
    import scipy

    path.mkdir(parents=True, exist_ok=True)

    # History log
    if graph._history:
        rows = list(graph._history)

        # normalize per-cell to avoid mixed dtypes at DataFrame construction
        def _norm(v):
            if v is None:
                return None
            if isinstance(v, np.generic):
                v = v.item()
            if hasattr(v, "to_list") and callable(getattr(v, "to_list", None)):
                try:
                    v = v.to_list()
                except Exception:
                    return str(v)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, set):
                try:
                    v = sorted(v)
                except Exception:
                    v = list(v)
            elif isinstance(v, tuple):
                v = list(v)
            try:
                from datetime import date as _dt_date
                from datetime import datetime as _dt_datetime

                if isinstance(v, (_dt_datetime, _dt_date)):
                    return v.isoformat()
            except Exception:
                pass
            if isinstance(v, (int, float, bool, str)):
                return v
            try:
                return json.dumps(v, default=str)
            except Exception:
                return str(v)

        keys = sorted({k for r in rows for k in r.keys()})
        cols = {k: [_norm(r.get(k)) for r in rows] for k in keys}

        # If polars exists, keep your typed polars path; otherwise pandas fallback.
        if _have_polars():
            import polars as pl

            def _dtype(vals):
                nonnull = [x for x in vals if x is not None]
                if nonnull and all(isinstance(x, bool) for x in nonnull):
                    return pl.Boolean
                if nonnull and all(isinstance(x, int) for x in nonnull):
                    return pl.Int64
                if nonnull and all(isinstance(x, (int, float)) for x in nonnull):
                    return pl.Float64
                return pl.Utf8

            schema_overrides = {k: _dtype(vs) for k, vs in cols.items()}
            history_df = pl.DataFrame(cols, schema_overrides=schema_overrides, strict=False)

            def is_nested(col: pl.Series) -> bool:
                sample = col.head(32).to_list()
                for v in sample:
                    if isinstance(v, (dict, list, set, tuple, np.ndarray)):
                        return True
                    try:
                        if hasattr(v, "to_list") and callable(v.to_list):
                            return True
                    except Exception:
                        pass
                return False

            nested_cols = [c for c in history_df.columns if is_nested(history_df[c])]

            def _jsonify_cell(v):
                if v is None:
                    return None
                if hasattr(v, "to_list") and callable(getattr(v, "to_list", None)):
                    try:
                        v = v.to_list()
                    except Exception:
                        v = str(v)
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                if isinstance(v, set):
                    try:
                        v = sorted(v)
                    except Exception:
                        v = list(v)
                elif isinstance(v, tuple):
                    v = list(v)
                try:
                    return json.dumps(v, default=str)
                except Exception:
                    return json.dumps(str(v))

            if nested_cols:
                history_df = history_df.with_columns(
                    *[
                        pl.col(c).map_elements(_jsonify_cell, return_dtype=pl.Utf8).alias(c)
                        for c in nested_cols
                    ]
                )

            history_df.write_parquet(path / "history.parquet", compression=compression)

        else:
            import pandas as pd

            history_df = pd.DataFrame(cols)

            # stringify any remaining nested/object-like cells
            def _jsonify_cell(v):
                if v is None:
                    return None
                if isinstance(v, (dict, list, set, tuple, np.ndarray)):
                    try:
                        if isinstance(v, np.ndarray):
                            v = v.tolist()
                        if isinstance(v, set):
                            v = sorted(v)
                        if isinstance(v, tuple):
                            v = list(v)
                        return json.dumps(v, default=str)
                    except Exception:
                        return str(v)
                return v

            # Only touch object dtype columns to avoid slowing numeric cols
            obj_cols = [c for c in history_df.columns if history_df[c].dtype == "object"]
            for c in obj_cols:
                history_df[c] = history_df[c].map(_jsonify_cell)

            _write_parquet_df(history_df, path / "history.parquet", compression=compression)

    # Provenance
    provenance = {
        "created": datetime.now(UTC).isoformat(),
        "annnet_version": "0.1.0",
        "python_version": sys.version,
        "dependencies": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "polars": (getattr(pl, "__version__", None) if pl is not None else None),
        },
    }
    (path / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # Snapshots directory (if any)
    (path / "snapshots").mkdir(exist_ok=True)


def _write_uns(graph, path: Path):
    """Write unstructured metadata and results."""
    import json

    path.mkdir(parents=True, exist_ok=True)

    # AnnNet attributes
    (path / "graph_attributes.json").write_text(
        json.dumps(graph.graph_attributes, indent=2, default=str)
    )

    # Results directory for algorithm outputs
    (path / "results").mkdir(exist_ok=True)


def read(path: str | Path, *, lazy: bool = False) -> AnnNet:
    """Load graph from disk with zero loss.

    Parameters
    ----------
    path : str | Path
        Path to .annnet directory
    lazy : bool, default False
        If True, delay loading large arrays until accessed

    Returns
    -------
    AnnNet
        Reconstructed graph with all topology and metadata

    """
    import json
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{path} not found")

    # FILE MODE (.annnet)
    if root.is_file() and root.suffix == ".annnet":
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = _read_archive(root, Path(tmp))
            return read(tmp_root, lazy=lazy)

    # 1. Read manifest
    manifest = json.loads((root / "manifest.json").read_text())

    # 2. Create empty graph
    from ..core.graph import AnnNet

    G = AnnNet(directed=manifest["directed"])
    G._version = manifest["graph_version"]

    # 3. Load structure
    _load_structure(G, root / "structure", lazy=lazy)

    # 4. Load tables
    _load_tables(G, root / "tables")

    # 5. Load layers (Kivela)
    _load_multilayers(G, root / "layers")

    # 6. Load slices
    _load_slices(G, root / "slices")

    # 7. Load audit
    _load_audit(G, root / "audit")

    # 8. Load uns
    _load_uns(G, root / "uns")

    # 9. Set active slice
    _current_slice = manifest["active_slice"]
    _default_slice = manifest["default_slice"]

    return G


def _load_structure(graph, path: Path, lazy: bool):
    """Load sparse matrix and index mappings."""
    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None
    import zarr

    # Load incidence matrix
    try:
        # Zarr v2
        inc_store = zarr.DirectoryStore(str(path / "incidence.zarr"))
        inc_root = zarr.group(store=inc_store)
    except AttributeError:
        # Zarr v3
        inc_root = zarr.open_group(str(path / "incidence.zarr"), mode="r")

    row = inc_root["row"][:]
    col = inc_root["col"][:]
    data = inc_root["data"][:]
    shape = tuple(inc_root.attrs["shape"])

    # Reconstruct as DOK for mutability
    coo = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    graph._matrix = coo.todok()

    # Load index mappings
    def parquet_to_dict(filepath: Path, key_col: str, val_col: str) -> dict:
        df = _read_parquet(filepath)
        return dict(zip(df[key_col], df[val_col]))

    graph.entity_to_idx = parquet_to_dict(path / "entity_to_idx.parquet", "entity_id", "idx")
    graph.idx_to_entity = parquet_to_dict(path / "idx_to_entity.parquet", "idx", "entity_id")
    graph.entity_types = parquet_to_dict(path / "entity_types.parquet", "entity_id", "type")
    graph.edge_to_idx = parquet_to_dict(path / "edge_to_idx.parquet", "edge_id", "idx")
    graph.idx_to_edge = parquet_to_dict(path / "idx_to_edge.parquet", "idx", "edge_id")
    graph.edge_weights = parquet_to_dict(path / "edge_weights.parquet", "edge_id", "weight")
    graph.edge_directed = parquet_to_dict(path / "edge_directed.parquet", "edge_id", "directed")
    graph.edge_kind = parquet_to_dict(path / "edge_kind.parquet", "edge_id", "kind")

    # Edge definitions
    edge_def_df = _read_parquet(path / "edge_definitions.parquet")
    graph.edge_definitions = {
        row["edge_id"]: (row["source"], row["target"], row["edge_type"])
        for row in _iter_rows(edge_def_df)
    }

    # Hyperedges
    hyper_path = path / "hyperedge_definitions.parquet"
    if hyper_path.exists():
        hyper_df = _read_parquet(hyper_path)
        graph.hyperedge_definitions = {}
        # Expect columns: edge_id, directed, members, head, tail
        for row in _iter_rows(hyper_df):
            eid = row["edge_id"]
            if row.get("directed", False):
                head = row.get("head")
                tail = row.get("tail")
                if head is None:
                    head = []
                if tail is None:
                    tail = []
                graph.hyperedge_definitions[eid] = {
                    "directed": True,
                    "head": head,
                    "tail": tail,
                }
            else:
                members = row.get("members")
                if members is None:
                    members = []
                graph.hyperedge_definitions[eid] = {
                    "directed": False,
                    "members": members,
                }

    # Update counts
    graph._num_entities = len(graph.entity_to_idx)
    graph._num_edges = len(graph.edge_to_idx)


def _load_tables(graph, path: Path):
    """Load Polars DataFrames."""
    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None

    graph.vertex_attributes = _read_parquet(path / "vertex_attributes.parquet")
    graph.edge_attributes = _read_parquet(path / "edge_attributes.parquet")
    graph.slice_attributes = _read_parquet(path / "slice_attributes.parquet")
    graph.edge_slice_attributes = _read_parquet(path / "edge_slice_attributes.parquet")


def _load_multilayers(graph, path: Path):
    """Load Kivela multilayer structures."""
    import json

    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None

    # Graceful exit if this is a legacy graph without layers
    if not path.exists() or not (path / "metadata.json").exists():
        return

    # 1. Metadata
    metadata = json.loads((path / "metadata.json").read_text())
    graph.aspects = metadata["aspects"]
    graph.elem_layers = metadata["elem_layers"]
    # [Vital] Rebuild the internal Cartesian product cache in the graph
    if hasattr(graph, "_rebuild_all_layers_cache"):
        graph._rebuild_all_layers_cache()

    # 2. Vertex Presence (V_M)
    if (path / "vertex_presence.parquet").exists():
        vm_df = _read_parquet(path / "vertex_presence.parquet")
        # Bulk update is faster than iterating if we can, but let's be safe
        for row in _iter_rows(vm_df):
            aa = tuple(row["layer"])  # Convert list back to tuple
            graph._VM.add((row["vertex_id"], aa))

    # 3. Edge Layers
    if (path / "edge_layers.parquet").exists():
        el_df = _read_parquet(path / "edge_layers.parquet")
        for row in _iter_rows(el_df):
            l1 = row["layer_1"]
            l2 = row["layer_2"]
            eid = row["edge_id"]
            if l2 is None:
                # Intra: stored as list, convert to tuple
                graph.edge_layers[eid] = tuple(l1)
            else:
                # Inter/Coupling: stored as two lists, convert to (tuple, tuple)
                graph.edge_layers[eid] = (tuple(l1), tuple(l2))

    # 4. Attributes

    # 4a. Elementary Layer Attributes (Load directly into Polars)
    if (path / "elem_layer_attributes.parquet").exists():
        graph.layer_attributes = _read_parquet(path / "elem_layer_attributes.parquet")

    # 4b. Aspect Attributes
    if (path / "aspect_attributes.json").exists():
        graph._aspect_attrs = json.loads((path / "aspect_attributes.json").read_text())

    # 4c. Tuple Layer Attributes
    if (path / "tuple_layer_attributes.parquet").exists():
        la_df = _read_parquet(path / "tuple_layer_attributes.parquet")
        for row in _iter_rows(la_df):
            aa = tuple(row["layer"])
            graph._layer_attrs[aa] = json.loads(row["attributes"])

    # 4d. Vertex-Layer Attributes
    if (path / "vertex_layer_attributes.parquet").exists():
        vla_df = _read_parquet(path / "vertex_layer_attributes.parquet")
        for row in _iter_rows(vla_df):
            key = (row["vertex_id"], tuple(row["layer"]))
            graph._vertex_layer_attrs[key] = json.loads(row["attributes"])


def _load_slices(graph, path: Path):
    """Reconstruct slice registry and memberships."""
    import json

    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None

    # Registry
    registry_df = _read_parquet(path / "registry.parquet")
    for row in _iter_rows(registry_df):
        slice_id = row["slice_id"]
        attrs = json.loads(row["attributes"])
        graph._slices[slice_id] = {"vertices": set(), "edges": set(), "attributes": attrs}

    # Vertex memberships
    vertex_df = _read_parquet(path / "vertex_memberships.parquet")
    for row in _iter_rows(vertex_df):
        graph._slices[row["slice_id"]]["vertices"].add(row["vertex_id"])

    # Edge memberships
    edge_df = _read_parquet(path / "edge_memberships.parquet")
    for row in _iter_rows(edge_df):
        lid = row["slice_id"]
        eid = row["edge_id"]
        graph._slices[lid]["edges"].add(eid)
        # Only set a per-slice weight if it was explicitly stored (not None).
        w = row.get("weight", None)
        if w is not None:
            graph.slice_edge_weights.setdefault(lid, {})[eid] = w


def _load_audit(graph, path: Path):
    """Load history and provenance."""
    try:
        import polars as pl  # optional
    except Exception:  # ModuleNotFoundError, etc.
        pl = None

    history_path = path / "history.parquet"
    if history_path.exists():
        history_df = _read_parquet(history_path)
        graph._history = _iter_rows(history_df)


def _load_uns(graph, path: Path):
    """Load unstructured metadata."""
    import json

    attrs_path = path / "graph_attributes.json"
    if attrs_path.exists():
        graph.graph_attributes = json.loads(attrs_path.read_text())
