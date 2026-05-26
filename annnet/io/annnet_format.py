"""
Native AnnNet archive format.

Provides:
    write(G, filename) -> None
    read(filename)     -> AnnNet

The native format stores graph structure, sparse matrices, dataframe-backed
tables, slice metadata, multilayer metadata, audit information, and unstructured
metadata in a compressed archive.

This module owns the on-disk AnnNet container format. It should use IO-local
helpers for archive handling and shared support helpers for dataframe and
serialization operations.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import UTC, datetime
import tempfile

import numpy as np
import scipy as scipy
import scipy.sparse as sp

from .. import __version__ as ANNNET_VERSION
from ._common import (
    dataframe_to_rows,
    dataframe_from_rows,
    dataframe_iter_rows,
    serialize_edge_layers,
    collect_slice_manifest,
    dataframe_from_columns,
    dataframe_read_parquet,
    restore_slice_manifest,
    dataframe_write_parquet,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)
from ._archive import _read_archive, _write_archive

if TYPE_CHECKING:
    from ..core import AnnNet


def _df_from_dict(data: dict):
    """Create a dataframe/table using AnnNet's configured backend."""
    if isinstance(data, list):
        return dataframe_from_rows(data)
    return dataframe_from_columns(data)


ANNNET_EXT = 'graph.annnet'


class _LayerDict:
    """Bidirectional int <-> layer-tuple mapping used by v2 parquet tables.

    Reserves id 0 for the placeholder layer ``('_',)`` so flat-graph rows
    that need a "no layer" sentinel land in a stable slot. Other layers get
    ids in first-seen order.
    """

    __slots__ = ('id_to_layer', 'layer_to_id')

    def __init__(self):
        self.id_to_layer: list[tuple] = []
        self.layer_to_id: dict[tuple, int] = {}
        self.intern(('_',))

    def intern(self, layer: tuple | None) -> int:
        if layer is None:
            return 0
        if not isinstance(layer, tuple):
            layer = tuple(layer)
        existing = self.layer_to_id.get(layer)
        if existing is not None:
            return existing
        new_id = len(self.id_to_layer)
        self.id_to_layer.append(layer)
        self.layer_to_id[layer] = new_id
        return new_id

    def get_id(self, layer: tuple | None) -> int | None:
        if layer is None:
            return None
        if not isinstance(layer, tuple):
            layer = tuple(layer)
        return self.layer_to_id.get(layer)

    def get_layer(self, layer_id: int | None) -> tuple | None:
        if layer_id is None:
            return None
        return self.id_to_layer[int(layer_id)]

    def to_json(self) -> dict:
        return {'layers': [list(t) for t in self.id_to_layer]}

    @classmethod
    def from_json(cls, data: dict) -> _LayerDict:
        self = cls.__new__(cls)
        self.id_to_layer = [tuple(layer) for layer in data.get('layers', [['_']])]
        if not self.id_to_layer:
            self.id_to_layer = [('_',)]
        self.layer_to_id = {layer: i for i, layer in enumerate(self.id_to_layer)}
        return self


def _build_layer_dict(graph) -> _LayerDict:
    """Walk graph state and intern every layer tuple that appears."""
    ld = _LayerDict()

    # 1. Entity index — the canonical source of layer coords
    for _vid, coord in graph._entities.keys():
        ld.intern(coord)

    # 2. Multilayer endpoints in binary edges
    for rec in graph._edges.values():
        for ep in (rec.src, rec.tgt):
            if (
                isinstance(ep, tuple)
                and len(ep) == 2
                and isinstance(ep[0], str)
                and isinstance(ep[1], tuple)
            ):
                ld.intern(ep[1])

    # 3. ml_layers stored on edges (intra: single tuple; inter/coupling: (a,b))
    for rec in graph._edges.values():
        ml = rec.ml_layers
        if ml is None:
            continue
        if isinstance(ml, tuple) and ml and isinstance(ml[0], tuple):
            ld.intern(ml[0])
            ld.intern(ml[1])
        elif isinstance(ml, tuple):
            ld.intern(ml)

    # 4. Per-(vertex, layer) attribute keys
    for key in getattr(graph, '_state_attrs', {}).keys():
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], tuple):
            ld.intern(key[1])

    # 5. Tuple-layer attribute keys
    try:
        for layer in graph.layers._layer_attrs.keys():
            ld.intern(layer)
    except AttributeError:
        pass

    return ld


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
        'created': datetime.now(UTC).isoformat(),
        'annnet_version': ANNNET_VERSION,
        'graph_version': graph._version,
        'directed': graph.directed,
        'counts': {
            'vertices': sum(1 for r in graph._entities.values() if r.kind == 'vertex'),
            'edges': graph.ne,
            'entities': len(graph._entities),
            'slices': len(graph.slices.list(include_default=True)),
            'hyperedges': len(graph.hyperedge_definitions),
            'aspects': len(graph.aspects),
        },
        'slices': list(graph.slices.list(include_default=True)),
        'active_slice': graph._current_slice,
        'default_slice': graph._default_slice,
        'compression': compression,
        # make encoding explicit for tests/docs
        'encoding': {'zarr': 'v3', 'parquet': '2.0'},
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # Build the layer dictionary once; every per-section writer that stores
    # layer coordinates references the same integer ids.
    layer_dict = _build_layer_dict(graph)

    # 2. Write structure/ (topology + layer dict)
    _write_structure(graph, root / 'structure', compression, layer_dict)

    # 3. Write tables/ (Polars > Parquet)
    _write_tables(graph, root / 'tables', compression)

    # 4. Write layers/ (Kivela Multilayer structures)
    _write_multilayers(graph, root / 'layers', compression, layer_dict)

    # 5. Write slices/
    _write_slices(graph, root / 'slices', compression, layer_dict)

    # 6. Write audit/
    _write_audit(graph, root / 'audit', compression)

    # 7. Write uns/
    _write_uns(graph, root / 'uns')


def write(
    graph: AnnNet,
    path: str | Path,
    *,
    compression: str = 'zstd',
    overwrite: bool = False,
) -> None:
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


def _write_structure(graph, path: Path, compression: str, layer_dict: _LayerDict):
    """Write sparse incidence matrix, layer dictionary, and merged edge tables.

    v2 layout:
      structure/
        incidence.zarr           — sparse incidence
        layer_dict.json          — id -> tuple[str, ...] mapping
        entity_index.parquet     — (entity_id, layer_id, idx, type)
        edges.parquet            — merged per-edge metadata + endpoints
        hyperedge_definitions.parquet — members / head / tail for hyperedges
    """
    import json

    import zarr

    path.mkdir(parents=True, exist_ok=True)

    # 1. Layer dictionary — read first on load, before any layer_id consumer
    (path / 'layer_dict.json').write_text(json.dumps(layer_dict.to_json()))

    # 2. Sparse incidence matrix (Zarr)
    coo = graph._matrix.tocoo()
    root = zarr.open_group(str(path / 'incidence.zarr'), mode='w')

    from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

    codec = BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)
    row = np.asarray(coo.row, dtype=np.int32)
    col = np.asarray(coo.col, dtype=np.int32)
    dat = np.asarray(coo.data, dtype=np.float32)
    root.create_array('row', data=row, chunks=(10000,), compressors=[codec])
    root.create_array('col', data=col, chunks=(10000,), compressors=[codec])
    root.create_array('data', data=dat, chunks=(10000,), compressors=[codec])
    root.attrs['shape'] = coo.shape

    # 3. Entity index — layer expressed as int32 id into layer_dict
    ent_ids: list = []
    ent_layer_ids: list = []
    ent_idxs: list = []
    ent_types: list = []
    for ekey, rec in graph._entities.items():
        ent_ids.append(ekey[0])
        ent_layer_ids.append(layer_dict.intern(ekey[1]))
        ent_idxs.append(rec.row_idx)
        ent_types.append(rec.kind)
    dataframe_write_parquet(
        dataframe_from_columns(
            {
                'entity_id': ent_ids,
                'layer_id': ent_layer_ids,
                'idx': ent_idxs,
                'type': ent_types,
            },
            schema={
                'entity_id': 'text',
                'layer_id': 'int',
                'idx': 'int',
                'type': 'text',
            },
        ),
        path / 'entity_index.parquet',
    )

    # 4. Merged edges.parquet — every edge gets one row
    #    Binary edges have source/target set; hyperedges have them null
    #    (their members live in hyperedge_definitions.parquet).
    #    Null-endpoint edge-entities also have source/target null.
    default_dir = True if graph.directed is None else graph.directed

    def _split_endpoint(ep):
        if (
            isinstance(ep, tuple)
            and len(ep) == 2
            and isinstance(ep[0], str)
            and isinstance(ep[1], tuple)
        ):
            return ep[0], layer_dict.intern(ep[1])
        if ep is None:
            return None, None
        return ep, None

    e_eids: list = []
    e_col_idxs: list = []
    e_weights: list = []
    e_directed: list = []
    e_kinds: list = []
    e_edge_types: list = []
    e_sources: list = []
    e_source_layer_ids: list = []
    e_targets: list = []
    e_target_layer_ids: list = []
    e_ml_kinds: list = []

    for eid, rec in graph._edges.items():
        e_eids.append(eid)
        e_col_idxs.append(int(rec.col_idx))
        e_weights.append(float(rec.weight) if rec.weight is not None else 1.0)
        e_directed.append(rec.directed)
        e_kinds.append(rec.etype)
        e_ml_kinds.append(rec.ml_kind)

        if rec.etype == 'hyper' or rec.src is None or isinstance(rec.src, (frozenset, set)):
            # Hyperedge or null placeholder; endpoints live elsewhere / are absent.
            e_sources.append(None)
            e_source_layer_ids.append(None)
            e_targets.append(None)
            e_target_layer_ids.append(None)
            if rec.etype != 'hyper':
                e_edge_types.append(None)
            else:
                e_edge_types.append(None)
        else:
            sid, slay = _split_endpoint(rec.src)
            tid, tlay = _split_endpoint(rec.tgt)
            e_sources.append(sid)
            e_source_layer_ids.append(slay)
            e_targets.append(tid)
            e_target_layer_ids.append(tlay)
            e_edge_types.append(
                'DIRECTED'
                if (rec.directed if rec.directed is not None else default_dir)
                else 'UNDIRECTED'
            )

    dataframe_write_parquet(
        dataframe_from_columns(
            {
                'edge_id': e_eids,
                'col_idx': e_col_idxs,
                'weight': e_weights,
                'directed': e_directed,
                'kind': e_kinds,
                'edge_type': e_edge_types,
                'source': e_sources,
                'source_layer_id': e_source_layer_ids,
                'target': e_targets,
                'target_layer_id': e_target_layer_ids,
                'ml_kind': e_ml_kinds,
            },
            schema={
                'edge_id': 'text',
                'col_idx': 'int',
                'weight': 'float',
                'directed': 'bool',
                'kind': 'text',
                'edge_type': 'text',
                'source': 'text',
                'source_layer_id': 'int',
                'target': 'text',
                'target_layer_id': 'int',
                'ml_kind': 'text',
            },
        ),
        path / 'edges.parquet',
    )

    # 5. Hyperedge definitions (members / head / tail). Same schema as v1.
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
        dataframe_write_parquet(hyper_df, path / 'hyperedge_definitions.parquet')


def _write_tables(graph, path: Path, compression: str):
    path.mkdir(parents=True, exist_ok=True)

    dataframe_write_parquet(graph.vertex_attributes, path / 'vertex_attributes.parquet')
    dataframe_write_parquet(graph.edge_attributes, path / 'edge_attributes.parquet')
    dataframe_write_parquet(graph.slice_attributes, path / 'slice_attributes.parquet')
    dataframe_write_parquet(graph.edge_slice_attributes, path / 'edge_slice_attributes.parquet')


def _write_multilayers(graph, path: Path, compression: str, layer_dict: _LayerDict):
    """Write Kivela multilayer structures with integer-encoded layer ids."""
    import json

    if not graph.aspects:
        return

    path.mkdir(parents=True, exist_ok=True)

    multilayer = serialize_multilayer_manifest(
        graph,
        table_to_rows=dataframe_to_rows,
        serialize_edge_layers=serialize_edge_layers,
    )

    # 1. Metadata: Aspects & elementary layer values
    (path / 'metadata.json').write_text(
        json.dumps(
            {
                'aspects': multilayer['aspects'],
                'elem_layers': multilayer['elem_layers'],
            },
            indent=2,
        )
    )

    # 2. Vertex presence: (vertex_id, layer_id)
    vm_vids: list = []
    vm_layer_ids: list = []
    for row in multilayer.get('VM', []):
        vm_vids.append(row.get('node') or row.get('vertex_id'))
        vm_layer_ids.append(layer_dict.intern(tuple(row.get('layer') or ())))
    dataframe_write_parquet(
        dataframe_from_columns(
            {'vertex_id': vm_vids, 'layer_id': vm_layer_ids},
            schema={'vertex_id': 'text', 'layer_id': 'int'},
        ),
        path / 'vertex_presence.parquet',
    )

    # 3. Edge layers: layer_1_id (always set), layer_2_id (set for inter/coupling).
    #    Drops the v1 edge_ml_kind.parquet — ml_kind is now in structure/edges.parquet.
    eids: list = []
    l1_ids: list = []
    l2_ids: list = []
    for eid, layers in deserialize_edge_layers(multilayer.get('edge_layers', {})).items():
        eids.append(eid)
        if isinstance(layers, tuple) and layers and isinstance(layers[0], tuple):
            l1_ids.append(layer_dict.intern(layers[0]))
            l2_ids.append(layer_dict.intern(layers[1]))
        else:
            l1_ids.append(layer_dict.intern(tuple(layers)))
            l2_ids.append(None)
    dataframe_write_parquet(
        dataframe_from_columns(
            {'edge_id': eids, 'layer_1_id': l1_ids, 'layer_2_id': l2_ids},
            schema={'edge_id': 'text', 'layer_1_id': 'int', 'layer_2_id': 'int'},
        ),
        path / 'edge_layers.parquet',
    )

    # 4a. Elementary layer attributes (table built upstream in graph.py)
    layer_attr_rows = multilayer.get('layer_attributes', [])
    if layer_attr_rows:
        dataframe_write_parquet(
            _df_from_dict(layer_attr_rows), path / 'elem_layer_attributes.parquet'
        )

    # 4b. Aspect attributes (small, JSON)
    if multilayer.get('aspect_attrs'):
        (path / 'aspect_attributes.json').write_text(
            json.dumps(multilayer['aspect_attrs'], indent=2)
        )

    # 4c. Tuple layer attributes: (layer_id, attributes_json)
    if multilayer.get('layer_tuple_attrs'):
        la_layer_ids: list = []
        la_attrs_json: list = []
        for row in multilayer['layer_tuple_attrs']:
            la_layer_ids.append(layer_dict.intern(tuple(row['layer'])))
            la_attrs_json.append(json.dumps(row.get('attrs') or row.get('attributes') or {}))
        dataframe_write_parquet(
            dataframe_from_columns(
                {'layer_id': la_layer_ids, 'attributes': la_attrs_json},
                schema={'layer_id': 'int', 'attributes': 'text'},
            ),
            path / 'tuple_layer_attributes.parquet',
        )

    # 4d. Vertex-layer attributes: (vertex_id, layer_id, attributes_json)
    if multilayer.get('node_layer_attrs'):
        vla_vids: list = []
        vla_layer_ids: list = []
        vla_attrs_json: list = []
        for row in multilayer['node_layer_attrs']:
            vla_vids.append(row.get('node') or row.get('vertex_id'))
            vla_layer_ids.append(layer_dict.intern(tuple(row.get('layer') or ())))
            vla_attrs_json.append(json.dumps(row.get('attrs') or row.get('attributes') or {}))
        dataframe_write_parquet(
            dataframe_from_columns(
                {
                    'vertex_id': vla_vids,
                    'layer_id': vla_layer_ids,
                    'attributes': vla_attrs_json,
                },
                schema={'vertex_id': 'text', 'layer_id': 'int', 'attributes': 'text'},
            ),
            path / 'vertex_layer_attributes.parquet',
        )


def _write_slices(graph, path: Path, compression: str, layer_dict: _LayerDict):
    """Write slice registry and memberships (multilayer-aware via layer_id)."""
    path.mkdir(parents=True, exist_ok=True)

    slice_membership, slice_weights = collect_slice_manifest(graph)

    # Registry: slice identifiers only. Slice attributes live in tables/.
    registry_data = []
    for slice_id in graph.slices.list(include_default=True):
        registry_data.append({'slice_id': slice_id})
    reg_df = _df_from_dict(registry_data)
    dataframe_write_parquet(reg_df, path / 'registry.parquet')

    # Vertex memberships: (slice_id, vertex_id, vertex_layer_id).
    # vertex ids may be bare strings or multilayer (vid, layer_coord) tuples;
    # encode layer as int id for a uniform parquet schema.
    def _split_vid(vid):
        if (
            isinstance(vid, tuple)
            and len(vid) == 2
            and isinstance(vid[0], str)
            and isinstance(vid[1], tuple)
        ):
            return vid[0], layer_dict.intern(vid[1])
        return vid, None

    s_slice_ids: list = []
    s_vids: list = []
    s_layer_ids: list = []
    for slice_id in graph.slices.list(include_default=True):
        for vertex_id in graph.slices.vertices(slice_id):
            bare, lid = _split_vid(vertex_id)
            s_slice_ids.append(slice_id)
            s_vids.append(bare)
            s_layer_ids.append(lid)
    vm_df = dataframe_from_columns(
        {'slice_id': s_slice_ids, 'vertex_id': s_vids, 'vertex_layer_id': s_layer_ids},
        schema={'slice_id': 'text', 'vertex_id': 'text', 'vertex_layer_id': 'int'},
    )
    dataframe_write_parquet(vm_df, path / 'vertex_memberships.parquet')

    # Edge memberships with weights
    edge_members: list[dict] = []
    for slice_id, edge_ids in slice_membership.items():
        for edge_id in edge_ids:
            edge_members.append(
                {
                    'slice_id': slice_id,
                    'edge_id': edge_id,
                    'weight': (slice_weights.get(slice_id) or {}).get(edge_id),
                }
            )
    # Ensure a stable schema even if there are zero rows
    if edge_members:
        em_df = _df_from_dict(edge_members)
    else:
        # explicit empty schema for pandas too
        em_df = _df_from_dict({'slice_id': [], 'edge_id': [], 'weight': []})
    dataframe_write_parquet(em_df, path / 'edge_memberships.parquet')


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
        dataframe_write_parquet(history_df, path / 'history.parquet')

    # Provenance
    try:
        from importlib import metadata as importlib_metadata

        polars_version = importlib_metadata.version('polars')
    except importlib_metadata.PackageNotFoundError:
        polars_version = None

    provenance = {
        'created': datetime.now(UTC).isoformat(),
        'annnet_version': ANNNET_VERSION,
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
    from ..core import AnnNet

    G = AnnNet(directed=manifest['directed'])
    G._version = manifest['graph_version']

    # Load the layer dictionary up front; both structure and multilayer
    # sections reference its integer ids.
    structure_dir = root / 'structure'
    layer_dict_path = structure_dir / 'layer_dict.json'
    if layer_dict_path.exists():
        layer_dict = _LayerDict.from_json(json.loads(layer_dict_path.read_text()))
    else:
        layer_dict = _LayerDict()

    # 3. Load structure
    _load_structure(G, structure_dir, lazy=lazy, layer_dict=layer_dict)

    # 4. Load tables
    _load_tables(G, root / 'tables')

    # 5. Load layers (Kivela)
    _load_multilayers(G, root / 'layers', layer_dict=layer_dict)

    # 6. Load slices
    _load_slices(G, root / 'slices', layer_dict=layer_dict)

    # 7. Load audit
    _load_audit(G, root / 'audit')

    # 8. Load uns
    _load_uns(G, root / 'uns')

    # 9. Set active slice
    G._current_slice = manifest['active_slice']
    G._default_slice = manifest['default_slice']

    return G


def _load_structure(graph, path: Path, lazy: bool, layer_dict: _LayerDict):
    """Load incidence matrix, entity index, and merged edges from v2 layout."""
    import zarr

    # 1. Sparse incidence matrix (Zarr)
    try:
        inc_store = zarr.DirectoryStore(str(path / 'incidence.zarr'))
        inc_root = zarr.group(store=inc_store)
    except AttributeError:
        inc_root = zarr.open_group(str(path / 'incidence.zarr'), mode='r')

    row = inc_root['row'][:]
    col = inc_root['col'][:]
    data = inc_root['data'][:]
    shape = tuple(inc_root.attrs['shape'])

    coo = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    graph._matrix = coo.todok()

    from annnet.core._records import EdgeRecord, EntityRecord

    # 2. Entity index — translate layer_id back to tuple coords
    graph._entities = {}
    graph._row_to_entity = {}
    for ent_row in dataframe_iter_rows(dataframe_read_parquet(path / 'entity_index.parquet')):
        layer_tuple = layer_dict.get_layer(ent_row['layer_id']) or ('_',)
        ekey = (ent_row['entity_id'], layer_tuple)
        rec = EntityRecord(row_idx=int(ent_row['idx']), kind=ent_row.get('type', 'vertex'))
        graph._entities[ekey] = rec
        graph._row_to_entity[rec.row_idx] = ekey
    graph._rebuild_entity_indexes()

    # 3. Edges — merged metadata + endpoint table
    graph._edges = {}
    graph._col_to_edge = {}

    def _reassemble_endpoint(vid, layer_id):
        if vid is None:
            return None
        if layer_id is None:
            return vid
        layer_tuple = layer_dict.get_layer(int(layer_id))
        return (vid, layer_tuple) if layer_tuple is not None else vid

    binary_specs: list = []  # (eid, col_idx)
    for er in dataframe_iter_rows(dataframe_read_parquet(path / 'edges.parquet')):
        eid = er['edge_id']
        col_idx = int(er['col_idx'])
        w = er.get('weight')
        weight = float(w) if w is not None else 1.0
        directed = er.get('directed')
        kind = er.get('kind') or 'binary'
        ml_kind = er.get('ml_kind')

        if kind == 'hyper':
            # Endpoints come from hyperedge_definitions.parquet; populate later.
            graph._edges[eid] = EdgeRecord(
                src=None,
                tgt=None,
                weight=weight,
                directed=directed,
                etype='hyper',
                col_idx=col_idx,
                ml_kind=ml_kind,
                ml_layers=None,
                direction_policy=None,
            )
        else:
            src = _reassemble_endpoint(er.get('source'), er.get('source_layer_id'))
            tgt = _reassemble_endpoint(er.get('target'), er.get('target_layer_id'))
            graph._edges[eid] = EdgeRecord(
                src=src,
                tgt=tgt,
                weight=weight,
                directed=directed,
                etype=kind,
                col_idx=col_idx,
                ml_kind=ml_kind,
                ml_layers=None,
                direction_policy=None,
            )
            if col_idx >= 0:
                binary_specs.append((eid, col_idx))
        if col_idx >= 0:
            graph._col_to_edge[col_idx] = eid

    # 4. Hyperedge member / head / tail
    hyper_path = path / 'hyperedge_definitions.parquet'
    if hyper_path.exists():
        for hrow in dataframe_iter_rows(dataframe_read_parquet(hyper_path)):
            eid = hrow['edge_id']
            rec = graph._edges.get(eid)
            if rec is None:
                continue
            is_dir = bool(hrow.get('directed', False))
            head = hrow.get('head') or []
            tail = hrow.get('tail') or []
            members = hrow.get('members') or []
            if is_dir:
                rec.src = frozenset(head)
                rec.tgt = frozenset(tail)
            else:
                rec.src = frozenset(members)
                rec.tgt = None
            rec.directed = is_dir

    # 5. Rebuild adjacency indices from _edges
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
    graph.vertex_attributes = dataframe_read_parquet(path / 'vertex_attributes.parquet')
    graph.edge_attributes = dataframe_read_parquet(path / 'edge_attributes.parquet')
    graph.slice_attributes = dataframe_read_parquet(path / 'slice_attributes.parquet')
    graph.edge_slice_attributes = dataframe_read_parquet(path / 'edge_slice_attributes.parquet')


def _load_multilayers(graph, path: Path, layer_dict: _LayerDict):
    """Load Kivela multilayer structures from v2 layout (int layer ids)."""
    import json

    if not path.exists() or not (path / 'metadata.json').exists():
        return

    legacy_flat_vertices = {
        vertex_id
        for (vertex_id, coord), rec in graph._entities.items()
        if rec.kind == 'vertex' and coord == ('_',)
    }

    metadata = json.loads((path / 'metadata.json').read_text())
    multilayer = {
        'aspects': metadata.get('aspects', []),
        'elem_layers': metadata.get('elem_layers', {}),
        'VM': [],
        'edge_kind': {},
        'edge_layers': {},
        'aspect_attrs': {},
        'node_layer_attrs': [],
        'layer_tuple_attrs': [],
        'layer_attributes': [],
    }

    def _layer_list(layer_id):
        t = layer_dict.get_layer(layer_id) if layer_id is not None else None
        return list(t) if t is not None else []

    if (path / 'vertex_presence.parquet').exists():
        vm_df = dataframe_read_parquet(path / 'vertex_presence.parquet')
        multilayer['VM'] = [
            {'node': row['vertex_id'], 'layer': _layer_list(row['layer_id'])}
            for row in dataframe_iter_rows(vm_df)
        ]

    if (path / 'edge_layers.parquet').exists():
        el_df = dataframe_read_parquet(path / 'edge_layers.parquet')
        raw = {}
        for row in dataframe_iter_rows(el_df):
            l1 = layer_dict.get_layer(row['layer_1_id'])
            l2_id = row.get('layer_2_id')
            if l2_id is None:
                raw[row['edge_id']] = l1
            else:
                raw[row['edge_id']] = (l1, layer_dict.get_layer(l2_id))
        multilayer['edge_layers'] = serialize_edge_layers(raw)

    # ml_kind ("intra"/"inter"/"coupling") was already loaded onto EdgeRecord
    # by _load_structure from edges.parquet — rebuild the manifest map here so
    # restore_multilayer_manifest can hand it back to the graph state machine.
    multilayer['edge_kind'] = {
        eid: rec.ml_kind for eid, rec in graph._edges.items() if rec.ml_kind is not None
    }

    if (path / 'elem_layer_attributes.parquet').exists():
        multilayer['layer_attributes'] = dataframe_to_rows(
            dataframe_read_parquet(path / 'elem_layer_attributes.parquet')
        )

    if (path / 'aspect_attributes.json').exists():
        multilayer['aspect_attrs'] = json.loads((path / 'aspect_attributes.json').read_text())

    if (path / 'tuple_layer_attributes.parquet').exists():
        la_df = dataframe_read_parquet(path / 'tuple_layer_attributes.parquet')
        multilayer['layer_tuple_attrs'] = [
            {
                'layer': _layer_list(row['layer_id']),
                'attrs': json.loads(row['attributes']),
            }
            for row in dataframe_iter_rows(la_df)
        ]

    if (path / 'vertex_layer_attributes.parquet').exists():
        vla_df = dataframe_read_parquet(path / 'vertex_layer_attributes.parquet')
        multilayer['node_layer_attrs'] = [
            {
                'node': row['vertex_id'],
                'layer': _layer_list(row['layer_id']),
                'attrs': json.loads(row['attributes']),
            }
            for row in dataframe_iter_rows(vla_df)
        ]

    restore_multilayer_manifest(
        graph,
        multilayer,
        rows_to_table=_df_from_dict,
        deserialize_edge_layers=deserialize_edge_layers,
    )

    # Native format roundtrips preserve the stored entity-index coordinates
    # exactly, even when multilayer metadata is declared afterwards.
    if legacy_flat_vertices and graph.aspects:
        placeholder = tuple('_' for _ in graph.aspects)
        remapped_entities = {}
        for (vertex_id, coord), rec in graph._entities.items():
            new_coord = (
                ('_',) if vertex_id in legacy_flat_vertices and coord == placeholder else coord
            )
            remapped_entities[(vertex_id, new_coord)] = rec
        graph._entities = remapped_entities
        if graph._state_attrs:
            graph._state_attrs = {
                (
                    vertex_id,
                    ('_',) if vertex_id in legacy_flat_vertices and coord == placeholder else coord,
                ): attrs
                for (vertex_id, coord), attrs in graph._state_attrs.items()
            }
        graph._rebuild_entity_indexes()


def _load_slices(graph, path: Path, layer_dict: _LayerDict):
    """Reconstruct slice registry and memberships from v2 layout."""
    registry_df = dataframe_read_parquet(path / 'registry.parquet')
    existing_slices = set(graph.slices.list(include_default=True))
    for row in dataframe_iter_rows(registry_df):
        slice_id = row['slice_id']
        if slice_id not in existing_slices:
            graph.slices.add(slice_id)
            existing_slices.add(slice_id)

    # Vertex memberships (vertex_layer_id may be null for bare-vid memberships)
    vertex_df = dataframe_read_parquet(path / 'vertex_memberships.parquet')
    for row in dataframe_iter_rows(vertex_df):
        bare = row['vertex_id']
        layer_id = row.get('vertex_layer_id')
        if layer_id is None:
            vertex_id = bare
        else:
            layer_tuple = layer_dict.get_layer(int(layer_id))
            vertex_id = (bare, layer_tuple) if layer_tuple is not None else bare
        if not graph.has_vertex(bare):
            continue
        graph.slices.add_vertex_to_slice(row['slice_id'], vertex_id)

    slice_membership: dict[str, list[str]] = {}
    slice_weights: dict[str, dict[str, float]] = {}
    edge_df = dataframe_read_parquet(path / 'edge_memberships.parquet')
    for row in dataframe_to_rows(edge_df):
        lid = row['slice_id']
        eid = row['edge_id']
        slice_membership.setdefault(lid, []).append(eid)
        w = row.get('weight', None)
        if w is not None:
            slice_weights.setdefault(lid, {})[eid] = w
    restore_slice_manifest(graph, slice_membership, slice_weights)


def _load_audit(graph, path: Path):
    """Load history and provenance."""
    history_path = path / 'history.parquet'
    if history_path.exists():
        history_df = dataframe_read_parquet(history_path)
        graph._history = dataframe_to_rows(history_df)


def _load_uns(graph, path: Path):
    """Load unstructured metadata."""
    import json

    attrs_path = path / 'graph_attributes.json'
    if attrs_path.exists():
        graph.graph_attributes = json.loads(attrs_path.read_text())
