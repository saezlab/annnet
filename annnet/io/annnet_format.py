from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import UTC, datetime
import tempfile

import numpy as np
import scipy as scipy
import scipy.sparse as sp

from ._utils import _read_archive, _write_archive
from ..adapters._utils import _finalize_multilayer_state
from .._support.dataframe_backend import (
    dataframe_to_rows,
    _dataframe_is_empty,
    dataframe_from_rows,
    dataframe_from_columns,
    _dataframe_read_parquet,
    _dataframe_write_parquet,
)

if TYPE_CHECKING:
    from ..core.graph import AnnNet


def _df_from_dict(data: dict):
    """Create a dataframe/table using AnnNet's configured backend."""
    if isinstance(data, list):
        return dataframe_from_rows(data)
    return dataframe_from_columns(data)


ANNNET_EXT = 'graph.annnet'


def _write_dir(graph, path: str | Path, *, compression='zstd', overwrite=False):
    """Write graph to disk with zero topology loss.

    Parameters
    ----------
    path : str | Path
        Target directory (e.g., "my_graph.annnet")
    compression : str, default "zstd"
        Compression codec for Zarr-backed arrays. Parquet tables use the
        configured dataframe backend defaults.
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
            raise FileExistsError(f'{root} already exists. Set overwrite=True.')
    else:
        root.mkdir(parents=True, exist_ok=True)

    # 1. Write manifest
    manifest = {
        'format': 'annnet',
        'version': '1.0.0',
        'created': datetime.now(UTC).isoformat(),
        'annnet_version': '0.1.0',
        'graph_version': graph._version,
        'directed': graph.directed,
        'counts': {
            'vertices': sum(1 for r in graph._entities.values() if r.kind == 'vertex'),
            'edges': len(graph._col_to_edge),
            'entities': len(graph._entities),
            'slices': len(graph._slices),
            'hyperedges': sum(1 for rec in graph._edges.values() if rec.etype == 'hyper'),
            'aspects': len(getattr(graph, 'aspects', [])),
        },
        'slices': list(graph._slices.keys()),
        'active_slice': graph._current_slice,
        'default_slice': graph._default_slice,
        'compression': compression,
        # make encoding explicit for tests/docs
        'encoding': {'zarr': 'v3', 'parquet': '2.0'},
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # 2. Write structure/ (topology)
    _write_structure(graph, root / 'structure', compression)

    # 3. Write tables/ (Polars > Parquet)
    _write_tables(graph, root / 'tables', compression)

    # 4. Write layers/ (Kivela Multilayer structures)
    _write_multilayers(graph, root / 'layers', compression)

    # 5. Write slices/
    _write_slices(graph, root / 'slices', compression)

    # 6. Write audit/
    _write_audit(graph, root / 'audit', compression)

    # 7. Write uns/
    _write_uns(graph, root / 'uns')


def write(graph, path: str | Path, *, compression='zstd', overwrite=False):
    """Write an AnnNet graph to a directory or `.annnet` archive."""
    path = Path(path)

    # FILE MODE (.annnet archive)
    if path.suffix == '.annnet' and not path.is_dir():
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} already exists. Set overwrite=True.')

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp) / 'graph.annnet'
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
    inc_path = path / 'incidence.zarr'

    # Zarr v3 compatibility
    import numpy as np

    root = zarr.open_group(str(inc_path), mode='w')

    from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

    codec = BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)

    row = np.asarray(coo.row, dtype=np.int32)
    col = np.asarray(coo.col, dtype=np.int32)
    dat = np.asarray(coo.data, dtype=np.float32)

    root.create_array('row', data=row, chunks=(10000,), compressors=[codec])
    root.create_array('col', data=col, chunks=(10000,), compressors=[codec])
    root.create_array('data', data=dat, chunks=(10000,), compressors=[codec])
    root.attrs['shape'] = coo.shape

    # Write all index mappings as Parquet
    def dict_to_parquet(d: dict, filepath: Path, id_name: str, val_name: str):
        df = _df_from_dict({id_name: list(d.keys()), val_name: list(d.values())})
        _dataframe_write_parquet(df, filepath)

    entity_rows = [
        {
            'entity_id': ekey[0],
            'layer': list(ekey[1]),
            'idx': rec.row_idx,
            'type': rec.kind,
        }
        for ekey, rec in graph._entities.items()
    ]
    _dataframe_write_parquet(_df_from_dict(entity_rows), path / 'entity_index.parquet')

    # Derive entity dicts from _entities / _row_to_entity.
    # These legacy tables remain for compatibility, while entity_index.parquet
    # preserves full multilayer coordinates.
    def _eid_str(eid):
        return eid[0] if isinstance(eid, tuple) else eid

    entity_to_idx = {_eid_str(eid): r.row_idx for eid, r in graph._entities.items()}
    idx_to_entity = {idx: _eid_str(eid) for idx, eid in graph._row_to_entity.items()}
    entity_types = {_eid_str(eid): r.kind for eid, r in graph._entities.items()}

    # Derive edge index dicts from _edges / _col_to_edge
    edge_to_idx = {eid: rec.col_idx for eid, rec in graph._edges.items() if rec.col_idx >= 0}
    idx_to_edge = dict(graph._col_to_edge)
    edge_weights = {eid: rec.weight for eid, rec in graph._edges.items()}
    edge_directed = {
        eid: rec.directed for eid, rec in graph._edges.items() if rec.directed is not None
    }
    edge_kind = {eid: rec.etype for eid, rec in graph._edges.items()}

    dict_to_parquet(entity_to_idx, path / 'entity_to_idx.parquet', 'entity_id', 'idx')
    dict_to_parquet(idx_to_entity, path / 'idx_to_entity.parquet', 'idx', 'entity_id')
    dict_to_parquet(entity_types, path / 'entity_types.parquet', 'entity_id', 'type')
    dict_to_parquet(edge_to_idx, path / 'edge_to_idx.parquet', 'edge_id', 'idx')
    dict_to_parquet(idx_to_edge, path / 'idx_to_edge.parquet', 'idx', 'edge_id')
    dict_to_parquet(edge_weights, path / 'edge_weights.parquet', 'edge_id', 'weight')
    dict_to_parquet(edge_directed, path / 'edge_directed.parquet', 'edge_id', 'directed')
    dict_to_parquet(edge_kind, path / 'edge_kind.parquet', 'edge_id', 'kind')

    # Edge definitions (binary + vertex_edge only; no hyper)
    bin_edges = {
        eid: rec
        for eid, rec in graph._edges.items()
        if rec.etype != 'hyper' and rec.src is not None
    }
    default_dir = True if graph.directed is None else graph.directed
    edge_def_df = _df_from_dict(
        {
            'edge_id': list(bin_edges.keys()),
            'source': [rec.src for rec in bin_edges.values()],
            'target': [rec.tgt for rec in bin_edges.values()],
            'edge_type': [
                'DIRECTED'
                if (rec.directed if rec.directed is not None else default_dir)
                else 'UNDIRECTED'
                for rec in bin_edges.values()
            ],
        }
    )
    _dataframe_write_parquet(edge_def_df, path / 'edge_definitions.parquet')

    # Hyperedge definitions
    hyper_edges = {eid: rec for eid, rec in graph._edges.items() if rec.etype == 'hyper'}
    if hyper_edges:
        eids, dirs, mems, heads, tails = [], [], [], [], []
        for eid, rec in hyper_edges.items():
            eids.append(eid)
            is_dir = rec.tgt is not None
            dirs.append(is_dir)
            if is_dir:
                heads.append(sorted(map(str, rec.src)))
                tails.append(sorted(map(str, rec.tgt)))
                mems.append(None)
            else:
                heads.append(None)
                tails.append(None)
                mems.append(sorted(map(str, rec.src)))

        hyper_df = _df_from_dict(
            {'edge_id': eids, 'directed': dirs, 'members': mems, 'head': heads, 'tail': tails}
        )
        _dataframe_write_parquet(hyper_df, path / 'hyperedge_definitions.parquet')


def _write_tables(graph, path: Path, compression: str):
    path.mkdir(parents=True, exist_ok=True)

    _dataframe_write_parquet(graph.vertex_attributes, path / 'vertex_attributes.parquet')
    _dataframe_write_parquet(graph.edge_attributes, path / 'edge_attributes.parquet')
    _dataframe_write_parquet(graph.slice_attributes, path / 'slice_attributes.parquet')
    _dataframe_write_parquet(graph.edge_slice_attributes, path / 'edge_slice_attributes.parquet')


def _write_multilayers(graph, path: Path, compression: str):
    """Write Kivela multilayer structures to disk."""
    import json

    # If no aspects are defined, skip creating the folder
    if not getattr(graph, 'aspects', []):
        return

    path.mkdir(parents=True, exist_ok=True)

    # 1. Metadata: Aspects & Elementary definitions
    metadata = {
        'aspects': graph.aspects,
        'elem_layers': graph.elem_layers,
    }
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    # 2. Vertex Presence (V_M)
    # Store "layer" as list[str] (works in polars; works in pandas+pyarrow as list column)
    vm_data = [{'vertex_id': u, 'layer': list(aa)} for u, aa in graph._VM]

    vm_df = _df_from_dict(vm_data if vm_data else {'vertex_id': [], 'layer': []})
    _dataframe_write_parquet(vm_df, path / 'vertex_presence.parquet')

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

    el_df = _df_from_dict({'edge_id': eids, 'layer_1': l1s, 'layer_2': l2s})
    _dataframe_write_parquet(el_df, path / 'edge_layers.parquet')

    # 3b. Multilayer edge kind ("intra"/"inter"/"coupling") — separate from structural etype
    if getattr(graph, 'edge_kind', None):
        ek_df = _df_from_dict(
            {'edge_id': list(graph.edge_kind.keys()), 'ml_kind': list(graph.edge_kind.values())}
        )
        _dataframe_write_parquet(ek_df, path / 'edge_ml_kind.parquet')

    # 4. Attributes (Specific Layer Stores)

    # 4a. Elementary Layer Attributes (Already a DataFrame in graph.py)
    if hasattr(graph, 'layer_attributes') and graph.layer_attributes is not None:
        if not _dataframe_is_empty(graph.layer_attributes):
            _dataframe_write_parquet(
                graph.layer_attributes,
                path / 'elem_layer_attributes.parquet',
            )

    # 4b. Aspect Attributes (Dict -> JSON)
    if hasattr(graph, '_aspect_attrs') and graph._aspect_attrs:
        (path / 'aspect_attributes.json').write_text(json.dumps(graph._aspect_attrs, indent=2))

    # 4c. Tuple Layer Attributes (Dict -> Parquet due to complex keys)
    if hasattr(graph, '_layer_attrs') and graph._layer_attrs:
        la_data = [
            {'layer': list(aa), 'attributes': json.dumps(attrs)}
            for aa, attrs in graph._layer_attrs.items()
        ]
        la_df = _df_from_dict(la_data)
        _dataframe_write_parquet(la_df, path / 'tuple_layer_attributes.parquet')

    # 4d. Vertex-Layer Attributes
    if hasattr(graph, '_state_attrs') and graph._state_attrs:
        vla_data = [
            {'vertex_id': u, 'layer': list(aa), 'attributes': json.dumps(attrs)}
            for (u, aa), attrs in graph._state_attrs.items()
        ]
        vla_df = _df_from_dict(vla_data)
        _dataframe_write_parquet(vla_df, path / 'vertex_layer_attributes.parquet')


def _write_slices(graph, path: Path, compression: str):
    """Write slice registry and memberships."""
    path.mkdir(parents=True, exist_ok=True)

    # Registry: slice identifiers only. Slice attributes live in tables/.
    registry_data = []
    for slice_id in graph._slices:
        registry_data.append({'slice_id': slice_id})
    reg_df = _df_from_dict(registry_data)
    _dataframe_write_parquet(reg_df, path / 'registry.parquet')

    # Vertex memberships: long format
    vertex_members = []
    for slice_id, slice_data in graph._slices.items():
        for vertex_id in slice_data['vertices']:
            vertex_members.append({'slice_id': slice_id, 'vertex_id': vertex_id})
    vm_df = _df_from_dict(vertex_members)
    _dataframe_write_parquet(vm_df, path / 'vertex_memberships.parquet')

    # Edge memberships with weights
    edge_members: list[dict] = []
    # Primary: explicit per-slice weights (if present)
    for slice_id, edge_weights in getattr(graph, 'slice_edge_weights', {}).items():
        for edge_id, weight in edge_weights.items():
            edge_members.append({'slice_id': slice_id, 'edge_id': edge_id, 'weight': weight})
    # Fallback: derive from registered slice edges if no explicit weights
    if not edge_members:
        for slice_id, slice_data in graph._slices.items():
            for edge_id in slice_data.get('edges', ()):
                edge_members.append({'slice_id': slice_id, 'edge_id': edge_id, 'weight': None})
    # Ensure a stable schema even if there are zero rows
    if edge_members:
        em_df = _df_from_dict(edge_members)
    else:
        # explicit empty schema for pandas too
        em_df = _df_from_dict({'slice_id': [], 'edge_id': [], 'weight': []})
    _dataframe_write_parquet(em_df, path / 'edge_memberships.parquet')


def _write_audit(graph, path: Path, compression: str):
    """Write history, snapshots, provenance."""
    import json

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
            if hasattr(v, 'to_list') and callable(getattr(v, 'to_list', None)):
                try:
                    v = v.to_list()
                except (AttributeError, TypeError, ValueError):
                    return str(v)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, set):
                try:
                    v = sorted(v)
                except TypeError:
                    v = list(v)
            elif isinstance(v, tuple):
                v = list(v)
            try:
                from datetime import (
                    date as _dt_date,
                    datetime as _dt_datetime,
                )

                if isinstance(v, (_dt_datetime, _dt_date)):
                    return v.isoformat()
            except ImportError:
                pass
            if isinstance(v, (int, float, bool, str)):
                return v
            try:
                return json.dumps(v, default=str)
            except (TypeError, ValueError):
                return str(v)

        keys = sorted({k for r in rows for k in r.keys()})
        cols = {k: [_norm(r.get(k)) for r in rows] for k in keys}

        history_df = dataframe_from_columns(cols)
        _dataframe_write_parquet(history_df, path / 'history.parquet')

    # Provenance
    try:
        from importlib import metadata as importlib_metadata

        polars_version = importlib_metadata.version('polars')
    except importlib_metadata.PackageNotFoundError:
        polars_version = None

    provenance = {
        'created': datetime.now(UTC).isoformat(),
        'annnet_version': '0.1.0',
        'python_version': sys.version,
        'dependencies': {
            'scipy': scipy.__version__,
            'numpy': np.__version__,
            'polars': polars_version,
        },
    }
    (path / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    # Snapshots directory (if any)
    (path / 'snapshots').mkdir(exist_ok=True)


def _write_uns(graph, path: Path):
    """Write unstructured metadata and results."""
    import json

    path.mkdir(parents=True, exist_ok=True)

    # AnnNet attributes
    (path / 'graph_attributes.json').write_text(
        json.dumps(graph.graph_attributes, indent=2, default=str)
    )

    # Results directory for algorithm outputs
    (path / 'results').mkdir(exist_ok=True)


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
        raise FileNotFoundError(f'{path} not found')

    # FILE MODE (.annnet)
    if root.is_file() and root.suffix == '.annnet':
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = _read_archive(root, Path(tmp))
            return read(tmp_root, lazy=lazy)

    # 1. Read manifest
    manifest = json.loads((root / 'manifest.json').read_text())

    # 2. Create empty graph
    from ..core.graph import AnnNet

    G = AnnNet(directed=manifest['directed'])
    G._version = manifest['graph_version']

    # 3. Load structure
    _load_structure(G, root / 'structure', lazy=lazy)

    # 4. Load tables
    _load_tables(G, root / 'tables')

    # 5. Load layers (Kivela)
    _load_multilayers(G, root / 'layers')

    # 6. Load slices
    _load_slices(G, root / 'slices')

    # 7. Load audit
    _load_audit(G, root / 'audit')

    # 8. Load uns
    _load_uns(G, root / 'uns')

    # 9. Set active slice
    G._current_slice = manifest['active_slice']
    G._default_slice = manifest['default_slice']

    return G


def _load_structure(graph, path: Path, lazy: bool):
    """Load sparse matrix and index mappings."""
    import zarr

    # Load incidence matrix
    try:
        # Zarr v2
        inc_store = zarr.DirectoryStore(str(path / 'incidence.zarr'))
        inc_root = zarr.group(store=inc_store)
    except AttributeError:
        # Zarr v3
        inc_root = zarr.open_group(str(path / 'incidence.zarr'), mode='r')

    row = inc_root['row'][:]
    col = inc_root['col'][:]
    data = inc_root['data'][:]
    shape = tuple(inc_root.attrs['shape'])

    # Reconstruct as DOK for mutability
    coo = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    graph._matrix = coo.todok()

    # Load index mappings
    def parquet_to_dict(filepath: Path, key_col: str, val_col: str) -> dict:
        df = _dataframe_read_parquet(filepath)
        return {row[key_col]: row[val_col] for row in dataframe_to_rows(df)}

    from annnet.core._records import EdgeRecord, EntityRecord

    # Build _entities and _row_to_entity from Parquet. Prefer the newer
    # entity_index.parquet which preserves multilayer coordinates exactly.
    entity_index_path = path / 'entity_index.parquet'
    if entity_index_path.exists():
        graph._entities = {}
        graph._row_to_entity = {}
        for row in dataframe_to_rows(_dataframe_read_parquet(entity_index_path)):
            ekey = (row['entity_id'], tuple(row.get('layer') or ['_']))
            rec = EntityRecord(row_idx=int(row['idx']), kind=row.get('type', 'vertex'))
            graph._entities[ekey] = rec
            graph._row_to_entity[rec.row_idx] = ekey
        graph._rebuild_entity_indexes()
    else:
        entity_to_idx = parquet_to_dict(path / 'entity_to_idx.parquet', 'entity_id', 'idx')
        entity_types = parquet_to_dict(path / 'entity_types.parquet', 'entity_id', 'type')
        raw_idx_to_entity = parquet_to_dict(path / 'idx_to_entity.parquet', 'idx', 'entity_id')
        graph._row_to_entity = {int(k): (v, ('_',)) for k, v in raw_idx_to_entity.items()}
        graph._entities = {
            (eid, ('_',)): EntityRecord(row_idx=int(idx), kind=entity_types.get(eid, 'vertex'))
            for eid, idx in entity_to_idx.items()
        }
        graph._rebuild_entity_indexes()

    # Load per-edge metadata for reconstruction
    edge_to_idx = parquet_to_dict(path / 'edge_to_idx.parquet', 'edge_id', 'idx')
    graph._col_to_edge = parquet_to_dict(path / 'idx_to_edge.parquet', 'idx', 'edge_id')
    edge_weights = parquet_to_dict(path / 'edge_weights.parquet', 'edge_id', 'weight')
    edge_directed = parquet_to_dict(path / 'edge_directed.parquet', 'edge_id', 'directed')
    edge_kind = parquet_to_dict(path / 'edge_kind.parquet', 'edge_id', 'kind')

    # Reconstruct _edges from edge_definitions + hyperedge_definitions
    graph._edges = {}

    edge_def_df = _dataframe_read_parquet(path / 'edge_definitions.parquet')
    for row in dataframe_to_rows(edge_def_df):
        eid = row['edge_id']
        etype = edge_kind.get(eid, 'binary')
        graph._edges[eid] = EdgeRecord(
            src=row['source'],
            tgt=row['target'],
            weight=float(edge_weights.get(eid, 1.0)),
            directed=edge_directed.get(eid),
            etype=etype,
            col_idx=int(edge_to_idx.get(eid, -1)),
            ml_kind=None,
            ml_layers=None,
            direction_policy=None,
        )

    hyper_path = path / 'hyperedge_definitions.parquet'
    if hyper_path.exists():
        hyper_df = _dataframe_read_parquet(hyper_path)
        for row in dataframe_to_rows(hyper_df):
            eid = row['edge_id']
            is_dir = bool(row.get('directed', False))
            head = row.get('head') or []
            tail = row.get('tail') or []
            members = row.get('members') or []
            if is_dir:
                src = frozenset(head)
                tgt = frozenset(tail)
            else:
                src = frozenset(members)
                tgt = None
            graph._edges[eid] = EdgeRecord(
                src=src,
                tgt=tgt,
                weight=float(edge_weights.get(eid, 1.0)),
                directed=is_dir,
                etype='hyper',
                col_idx=int(edge_to_idx.get(eid, -1)),
                ml_kind=None,
                ml_layers=None,
                direction_policy=None,
            )

    # Rebuild adjacency indices from _edges
    graph._adj = {}
    graph._src_to_edges = {}
    graph._tgt_to_edges = {}
    for eid, rec in graph._edges.items():
        if rec.etype != 'hyper' and rec.src is not None and rec.tgt is not None:
            key = (rec.src, rec.tgt)
            graph._adj.setdefault(key, []).append(eid)
            graph._src_to_edges.setdefault(rec.src, []).append(eid)
            graph._tgt_to_edges.setdefault(rec.tgt, []).append(eid)


def _load_tables(graph, path: Path):
    """Load annotation tables with the configured dataframe backend."""
    graph.vertex_attributes = _dataframe_read_parquet(path / 'vertex_attributes.parquet')
    graph.edge_attributes = _dataframe_read_parquet(path / 'edge_attributes.parquet')
    graph.slice_attributes = _dataframe_read_parquet(path / 'slice_attributes.parquet')
    graph.edge_slice_attributes = _dataframe_read_parquet(path / 'edge_slice_attributes.parquet')


def _load_multilayers(graph, path: Path):
    """Load Kivela multilayer structures."""
    import json

    # Graceful exit if this is a legacy graph without layers
    if not path.exists() or not (path / 'metadata.json').exists():
        return

    # 1. Metadata
    metadata = json.loads((path / 'metadata.json').read_text())
    _finalize_multilayer_state(graph, metadata['aspects'], metadata['elem_layers'])

    # 2. Vertex Presence (V_M)
    if (path / 'vertex_presence.parquet').exists():
        vm_df = _dataframe_read_parquet(path / 'vertex_presence.parquet')
        vm_set = {(row['vertex_id'], tuple(row['layer'])) for row in dataframe_to_rows(vm_df)}
        if vm_set:
            graph._restore_supra_nodes(vm_set)
            graph._VM = vm_set

    # 3. Edge Layers
    if (path / 'edge_layers.parquet').exists():
        el_df = _dataframe_read_parquet(path / 'edge_layers.parquet')
        for row in dataframe_to_rows(el_df):
            l1 = row['layer_1']
            l2 = row['layer_2']
            eid = row['edge_id']
            if l2 is None:
                # Intra: stored as list, convert to tuple
                graph.edge_layers[eid] = tuple(l1)
            else:
                # Inter/Coupling: stored as two lists, convert to (tuple, tuple)
                graph.edge_layers[eid] = (tuple(l1), tuple(l2))

    # 3b. Multilayer edge kind
    if (path / 'edge_ml_kind.parquet').exists():
        ek_df = _dataframe_read_parquet(path / 'edge_ml_kind.parquet')
        for row in dataframe_to_rows(ek_df):
            graph.edge_kind[row['edge_id']] = row['ml_kind']

    # 4. Attributes

    # 4a. Elementary Layer Attributes (Load directly into Polars)
    if (path / 'elem_layer_attributes.parquet').exists():
        graph.layer_attributes = _dataframe_read_parquet(path / 'elem_layer_attributes.parquet')

    # 4b. Aspect Attributes
    if (path / 'aspect_attributes.json').exists():
        graph._aspect_attrs = json.loads((path / 'aspect_attributes.json').read_text())

    # 4c. Tuple Layer Attributes
    if (path / 'tuple_layer_attributes.parquet').exists():
        la_df = _dataframe_read_parquet(path / 'tuple_layer_attributes.parquet')
        for row in dataframe_to_rows(la_df):
            aa = tuple(row['layer'])
            graph._layer_attrs[aa] = json.loads(row['attributes'])

    # 4d. Vertex-Layer Attributes
    if (path / 'vertex_layer_attributes.parquet').exists():
        vla_df = _dataframe_read_parquet(path / 'vertex_layer_attributes.parquet')
        for row in dataframe_to_rows(vla_df):
            key = (row['vertex_id'], tuple(row['layer']))
            graph._state_attrs[key] = json.loads(row['attributes'])


def _load_slices(graph, path: Path):
    """Reconstruct slice registry and memberships."""
    from ..core._records import SliceRecord

    # Registry
    registry_df = _dataframe_read_parquet(path / 'registry.parquet')
    for row in dataframe_to_rows(registry_df):
        slice_id = row['slice_id']
        graph._slices[slice_id] = SliceRecord()

    # Vertex memberships
    vertex_df = _dataframe_read_parquet(path / 'vertex_memberships.parquet')
    for row in dataframe_to_rows(vertex_df):
        graph._slices[row['slice_id']]['vertices'].add(row['vertex_id'])

    # Edge memberships
    edge_df = _dataframe_read_parquet(path / 'edge_memberships.parquet')
    for row in dataframe_to_rows(edge_df):
        lid = row['slice_id']
        eid = row['edge_id']
        graph._slices[lid]['edges'].add(eid)
        # Only set a per-slice weight if it was explicitly stored (not None).
        w = row.get('weight', None)
        if w is not None:
            graph.slice_edge_weights.setdefault(lid, {})[eid] = w


def _load_audit(graph, path: Path):
    """Load history and provenance."""
    history_path = path / 'history.parquet'
    if history_path.exists():
        history_df = _dataframe_read_parquet(history_path)
        graph._history = dataframe_to_rows(history_df)


def _load_uns(graph, path: Path):
    """Load unstructured metadata."""
    import json

    attrs_path = path / 'graph_attributes.json'
    if attrs_path.exists():
        graph.graph_attributes = json.loads(attrs_path.read_text())
