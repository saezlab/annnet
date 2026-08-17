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
from ..core import _structure
from ._common import (
    LOADED_EDGE_KIND,
    STORED_EDGE_KIND,
    LOADED_ENTITY_KIND,
    STORED_ENTITY_KIND,
    dataframe_columns,
    dataframe_to_rows,
    serialize_endpoint,
    dataframe_from_rows,
    dataframe_iter_rows,
    deserialize_endpoint,
    serialize_edge_layers,
    collect_slice_manifest,
    dataframe_from_columns,
    dataframe_read_parquet,
    dataframe_write_parquet,
    deserialize_edge_layers,
    rename_dataframe_columns,
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


def _serialize_hyper_members(members) -> list[str]:
    """Encode hyperedge members deterministically for parquet round-trip."""
    import json

    return [
        json.dumps(serialize_endpoint(member), sort_keys=True)
        for member in sorted(members, key=repr)
    ]


def _deserialize_hyper_members(members) -> list:
    """Restore hyperedge members from parquet-safe endpoint payloads."""
    return [deserialize_endpoint(member) for member in (members or [])]


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
    for ref in _structure.iter_entities(graph):
        ld.intern(ref.layer)

    # 2. Multilayer endpoints in binary edges
    for edge in _structure.iter_edges(graph, include_placeholders=True):
        sides = _structure.edge_sides(graph, edge.id)
        for endpoint in sides.source | sides.target:
            if _structure.is_entity_key(endpoint):
                ld.intern(endpoint[1])

    # 3. ml_layers stored on edges (intra: single tuple; inter/coupling: (a,b))
    for edge in _structure.iter_edges(graph, include_placeholders=True):
        ml = edge.ml_layers
        if ml is None:
            continue
        if isinstance(ml, tuple) and ml and isinstance(ml[0], tuple):
            ld.intern(ml[0])
            ld.intern(ml[1])
        elif isinstance(ml, tuple):
            ld.intern(ml)

    # 4. Per-(node, layer) attribute keys
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


def _write_dir(graph, path: str | Path, *, compression='zstd', overwrite=False, matrix=False):
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
    matrix : bool, default False
        Also persist the incidence matrix. It is a derived cache — the tables
        written here, the coefficient table among them, fully reconstruct it —
        so the default omits it and lets ``read()`` rebuild lazily. Pass True to
        trade file size for skipping that rebuild.

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
    # The incidence matrix is a derived cache: every column is reconstructable from
    # the records, since explicit coeffs now persist as records data in
    # edge_coeffs.parquet. Storing it is therefore opt-in — a size/load-time trade,
    # never a correctness one. The shape is recorded either way so a lazy rebuild
    # knows its extent.
    incidence_stored = bool(matrix)
    manifest = {
        'format': 'annnet',
        'created': datetime.now(UTC).isoformat(),
        'annnet_version': ANNNET_VERSION,
        'graph_version': graph._version,
        'directed': graph.directed,
        'matrix_shape': list(graph.S.shape),
        'incidence_stored': incidence_stored,
        'counts': {
            'nodes': sum(
                1 for ref in _structure.iter_entities(graph) if ref.kind == _structure.NODE
            ),
            'edges': graph.ne,
            'entities': sum(1 for _ in _structure.iter_entities(graph)),
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
    _write_structure(
        graph, root / 'structure', compression, layer_dict, store_incidence=incidence_stored
    )

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
    matrix: bool = False,
) -> None:
    """Write an AnnNet graph to a directory or `.annnet` archive.

    Parameters
    ----------
    matrix : bool, default False
        Also persist the incidence matrix. The records are the source of truth and
        fully reconstruct it, so this is a size/load-time trade rather than a
        correctness one: with it omitted, ``read()`` defers a rebuild until the
        matrix is first touched. Explicit coefficients persist either way, as
        records data.
    """
    path = Path(path)

    # FILE MODE (.annnet archive)
    if path.suffix == '.annnet' and not path.is_dir():
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} already exists. Set overwrite=True.')

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp) / 'graph.annnet'
            _write_dir(graph, tmp_root, compression=compression, overwrite=True, matrix=matrix)
            _write_archive(tmp_root, path)
        return

    # DIRECTORY MODE (canonical format)
    return _write_dir(graph, path, compression=compression, overwrite=overwrite, matrix=matrix)


def _write_structure(
    graph, path: Path, compression: str, layer_dict: _LayerDict, *, store_incidence: bool = True
):
    """Write sparse incidence matrix, layer dictionary, and merged edge tables.

    v2 layout:
      structure/
        incidence.zarr           — sparse incidence
        layer_dict.json          — id -> tuple[str, ...] mapping
        entity_index.parquet     — (entity_id, layer_id, idx, type)
        edges.parquet            — merged per-edge metadata + endpoints
        hyperedge_definitions.parquet — members / head / tail for hyperedges
        edge_direction_policy.json    — edge_id -> the policy, for the edges with one
    """
    import json

    import zarr

    path.mkdir(parents=True, exist_ok=True)

    # 1. Layer dictionary — read first on load, before any layer_id consumer
    (path / 'layer_dict.json').write_text(json.dumps(layer_dict.to_json()))

    # A flexible-direction policy says which way an edge points, and it is
    # declared rather than derived, so a file that does not carry it loses what
    # the user set. Almost no edge has one, so the file is written only when some
    # edge does and a reader that finds none takes that as none.
    policies = _structure.edge_policies(graph)
    if policies:
        (path / 'edge_direction_policy.json').write_text(json.dumps(policies, indent=2))

    # 2. Sparse incidence matrix (Zarr) — only when it holds explicit coeffs;
    # otherwise it is rebuilt from records on load (see read()).
    if store_incidence:
        coo = graph.S.tocoo()
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
    for ref in _structure.iter_entities(graph):
        ent_ids.append(ref.id)
        ent_layer_ids.append(layer_dict.intern(ref.layer))
        ent_idxs.append(_structure.entity_row(graph, ref.key))
        ent_types.append(STORED_ENTITY_KIND.get(ref.kind, ref.kind))
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
    # Hyperedge layer tuple — single layer for all members per the layer=
    # convention on _add_hyperedges_batch. Binary edges reconstruct
    # ml_layers from (source_layer_id, target_layer_id) on load and leave
    # this slot null.
    e_hyper_layer_ids: list = []

    for edge in _structure.iter_edges(graph, include_placeholders=True):
        sides = _structure.edge_sides(graph, edge.id)
        source, target = sides.source, sides.target
        is_hyper = edge.kind == _structure.HYPER

        e_eids.append(edge.id)
        e_col_idxs.append(_structure.edge_column(graph, edge.id))
        e_weights.append(float(edge.weight))
        e_directed.append(edge.declared_directed)
        e_kinds.append(STORED_EDGE_KIND.get(edge.kind, edge.kind))
        e_ml_kinds.append(edge.ml_kind)

        if is_hyper or not source or len(source) > 1 or len(target) > 1:
            # Hyperedge or null placeholder; endpoints live elsewhere / are absent.
            e_sources.append(None)
            e_source_layer_ids.append(None)
            e_targets.append(None)
            e_target_layer_ids.append(None)
            e_edge_types.append(None)
            if (
                is_hyper
                and isinstance(edge.ml_layers, tuple)
                and edge.ml_layers
                and isinstance(edge.ml_layers[0], str)
            ):
                e_hyper_layer_ids.append(layer_dict.intern(edge.ml_layers))
            else:
                e_hyper_layer_ids.append(None)
        else:
            sid, slay = _split_endpoint(next(iter(source), None))
            tid, tlay = _split_endpoint(next(iter(target), None))
            e_sources.append(sid)
            e_source_layer_ids.append(slay)
            e_targets.append(tid)
            e_target_layer_ids.append(tlay)
            e_edge_types.append(
                'DIRECTED'
                if (edge.declared_directed if edge.declared_directed is not None else default_dir)
                else 'UNDIRECTED'
            )
            e_hyper_layer_ids.append(None)

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
                'hyper_layer_id': e_hyper_layer_ids,
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
                'hyper_layer_id': 'int',
            },
        ),
        path / 'edges.parquet',
    )

    # 4b. Explicit incidence coefficients (stoichiometry, resolved flexible
    # directions). An edge that carries these cannot derive its column from its
    # weight and its directedness, so they are what makes the file a complete
    # description, and they are written whether or not the derived matrix is.
    # Long format: one row per (edge, node), keyed exactly like source/target.
    # Float64 here, where the matrix is float32; the stored value survives intact.
    c_eids: list = []
    c_entities: list = []
    c_layer_ids: list = []
    c_values: list = []
    for edge in _structure.iter_edges(graph, include_placeholders=True):
        coefficients = _structure.edge_coefficients(graph, edge.id)
        if not coefficients:
            continue
        eid = edge.id
        # Sorted so the table is byte-stable across processes: a coefficient map is
        # keyed by node, and iteration over strings follows randomised hash order.
        for node, value in sorted(coefficients.items(), key=lambda kv: repr(kv[0])):
            vid, layer_id = _split_endpoint(node)
            c_eids.append(eid)
            c_entities.append(vid)
            c_layer_ids.append(layer_id)
            c_values.append(float(value))
    if c_eids:
        dataframe_write_parquet(
            dataframe_from_columns(
                {
                    'edge_id': c_eids,
                    'entity_id': c_entities,
                    'layer_id': c_layer_ids,
                    'coeff': c_values,
                },
                schema={
                    'edge_id': 'text',
                    'entity_id': 'text',
                    'layer_id': 'int',
                    'coeff': 'float',
                },
            ),
            path / 'edge_coeffs.parquet',
        )

    # 5. Hyperedge definitions (members / head / tail). Same schema as v1.
    hyper_edges = [
        edge
        for edge in _structure.iter_edges(graph, include_placeholders=True)
        if edge.kind == _structure.HYPER
    ]
    if hyper_edges:
        eids, dirs, mems, heads, tails = [], [], [], [], []
        for edge in hyper_edges:
            sides = _structure.edge_sides(graph, edge.id)
            eids.append(edge.id)
            is_dir = bool(sides.target)
            dirs.append(is_dir)
            if is_dir:
                heads.append(_serialize_hyper_members(sides.source))
                tails.append(_serialize_hyper_members(sides.target))
                mems.append(None)
            else:
                heads.append(None)
                tails.append(None)
                mems.append(_serialize_hyper_members(sides.source))
        hyper_df = _df_from_dict(
            {'edge_id': eids, 'directed': dirs, 'members': mems, 'head': heads, 'tail': tails}
        )
        dataframe_write_parquet(hyper_df, path / 'hyperedge_definitions.parquet')


def _write_tables(graph, path: Path, compression: str):
    path.mkdir(parents=True, exist_ok=True)

    dataframe_write_parquet(graph._node_table, path / '_node_table.parquet')
    dataframe_write_parquet(graph._edge_table, path / '_edge_table.parquet')
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
        # native writer reads graph.edge_layers directly below; skip the
        # serialize → deserialize round-trip the manifest used to do.
        include_edge_layers=False,
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

    # 2. Node presence: (node_id, layer_id)
    vm_vids: list = []
    vm_layer_ids: list = []
    for row in multilayer.get('VM', []):
        vm_vids.append(row.get('node') or row.get('node_id'))
        vm_layer_ids.append(layer_dict.intern(tuple(row.get('layer') or ())))
    dataframe_write_parquet(
        dataframe_from_columns(
            {'node_id': vm_vids, 'layer_id': vm_layer_ids},
            schema={'node_id': 'text', 'layer_id': 'int'},
        ),
        path / 'node_presence.parquet',
    )

    # 3. Edge layers: layer_1_id (always set), layer_2_id (set for inter/coupling).
    #    Drops the v1 edge_ml_kind.parquet — ml_kind is now in structure/edges.parquet.
    #
    # Read raw graph.edge_layers directly to avoid the serialize→deserialize
    # round-trip that previously cost ~0.3s on UC2-scale graphs.
    eids: list = []
    l1_ids: list = []
    l2_ids: list = []
    for eid, layers in graph.edge_layers.items():
        if layers is None:
            continue
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

    # 4d. Node-layer attributes: (node_id, layer_id, attributes_json)
    if multilayer.get('node_layer_attrs'):
        vla_vids: list = []
        vla_layer_ids: list = []
        vla_attrs_json: list = []
        for row in multilayer['node_layer_attrs']:
            vla_vids.append(row.get('node') or row.get('node_id'))
            vla_layer_ids.append(layer_dict.intern(tuple(row.get('layer') or ())))
            vla_attrs_json.append(json.dumps(row.get('attrs') or row.get('attributes') or {}))
        dataframe_write_parquet(
            dataframe_from_columns(
                {
                    'node_id': vla_vids,
                    'layer_id': vla_layer_ids,
                    'attributes': vla_attrs_json,
                },
                schema={'node_id': 'text', 'layer_id': 'int', 'attributes': 'text'},
            ),
            path / 'node_layer_attributes.parquet',
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

    # Node memberships: (slice_id, node_id, node_layer_id).
    # node ids may be bare strings or multilayer (vid, layer_coord) tuples;
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
        for node_id in graph.slices.nodes(slice_id):
            bare, lid = _split_vid(node_id)
            s_slice_ids.append(slice_id)
            s_vids.append(bare)
            s_layer_ids.append(lid)
    vm_df = dataframe_from_columns(
        {'slice_id': s_slice_ids, 'node_id': s_vids, 'node_layer_id': s_layer_ids},
        schema={'slice_id': 'text', 'node_id': 'text', 'node_layer_id': 'int'},
    )
    dataframe_write_parquet(vm_df, path / 'node_memberships.parquet')

    # Edge memberships with weights
    # Columnar (mirrors node memberships): building one dict per membership
    # row and transposing dominated write() on the edge-membership table.
    em_slice_ids: list = []
    em_edge_ids: list = []
    em_weights: list = []
    for slice_id, edge_ids in slice_membership.items():
        sw = slice_weights.get(slice_id) or {}
        for edge_id in edge_ids:
            em_slice_ids.append(slice_id)
            em_edge_ids.append(edge_id)
            em_weights.append(sw.get(edge_id))
    em_df = dataframe_from_columns(
        {'slice_id': em_slice_ids, 'edge_id': em_edge_ids, 'weight': em_weights},
        schema={'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
    )
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

    # Snapshots directory + snapshots.json (preserves the list returned by
    # G.history.list_snapshots()). Sets are serialized as sorted lists.
    (path / 'snapshots').mkdir(exist_ok=True)
    snaps = getattr(graph, '_snapshots', None)
    if snaps:
        serialised = []
        for snap in snaps:
            serialised.append(
                {
                    'label': snap.get('label'),
                    'version': snap.get('version'),
                    'node_ids': sorted(snap.get('node_ids', set())),
                    'edge_ids': sorted(snap.get('edge_ids', set())),
                    'slice_ids': sorted(snap.get('slice_ids', set())),
                }
            )
        (path / 'snapshots.json').write_text(json.dumps(serialised))


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

    # Declare aspects BEFORE loading entities so _load_structure can write
    # entities directly at their stored layer coords. If we let
    # _load_multilayers call set_aspects after _load_structure, set_aspects
    # would see the populated _entities, treat them as flat, and remap them
    # all to the placeholder layer — losing the saved layer assignment.
    layers_dir = root / 'layers'
    if layers_dir.exists() and (layers_dir / 'metadata.json').exists():
        multilayer_meta = json.loads((layers_dir / 'metadata.json').read_text())
        meta_aspects = multilayer_meta.get('aspects') or []
        meta_elem = multilayer_meta.get('elem_layers') or {}
        if meta_aspects:
            G.layers.set_aspects(meta_aspects)
            if meta_elem:
                G.layers.set_elementary_layers(meta_elem)
            # Pre-register the '_' placeholder per aspect so any entity
            # stored at a placeholder coord (legacy or current) validates
            # cleanly when `_make_layer_coord` runs on it later.
            G._ensure_placeholder_layers_declared()

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


def _columns_as_lists(df, names):
    """Return ``{name: list}`` for each requested column (missing -> all None).

    Columnar access avoids materialising one dict per row (polars ``iter_rows``
    dominated ``read()`` on the edges table); the loop then ``zip``s the lists.
    """
    try:
        import polars as _pl

        if isinstance(df, _pl.DataFrame):
            height = df.height
            present = set(df.columns)
            return {
                name: (df[name].to_list() if name in present else [None] * height) for name in names
            }
    except ImportError:
        pass
    rows = list(dataframe_iter_rows(df))
    return {name: [r.get(name) for r in rows] for name in names}


def _load_structure(graph, path: Path, lazy: bool, layer_dict: _LayerDict):
    """Load incidence matrix, entity index, and merged edges from v2 layout."""
    import zarr

    # 1. Sparse incidence matrix. The graph derives its own from the store, so
    # this is not installed anywhere. It is read because a file written before
    # the coefficient table existed records an explicit coefficient nowhere
    # else, and step 7 recovers those from it.
    if (path / 'incidence.zarr').exists():
        try:
            inc_store = zarr.DirectoryStore(str(path / 'incidence.zarr'))
            inc_root = zarr.group(store=inc_store)
        except AttributeError:
            inc_root = zarr.open_group(str(path / 'incidence.zarr'), mode='r')

        row = inc_root['row'][:]
        col = inc_root['col'][:]
        data = inc_root['data'][:]
        shape = tuple(inc_root.attrs['shape'])

        # A column is an edge, so the recovery reads this by column.
        matrix = sp.coo_array((data, (row, col)), shape=shape, dtype=np.float32).tocsc()
    else:
        matrix = None

    # 2. Entity index — translate layer_id back to tuple coords.
    # Read columns once and iterate via zip over arrays instead of yielding one
    # dict per row. At UC1 scale (~1.87M supra-nodes) the previous code spent
    # most of read() on this loop.
    ent_df = dataframe_read_parquet(path / 'entity_index.parquet')
    try:
        # Polars fast path — direct column access, no per-row dicts.
        import polars as _pl

        if isinstance(ent_df, _pl.DataFrame):
            ent_ids_col = ent_df['entity_id'].to_list()
            ent_layer_col = ent_df['layer_id'].to_list()
            ent_idx_col = ent_df['idx'].to_list()
            ent_type_col = ent_df['type'].to_list() if 'type' in ent_df.columns else None
        else:
            raise TypeError('not polars')
    except (ImportError, TypeError):
        # Generic backend: fall back to a single iter_rows pass.
        ent_ids_col, ent_layer_col, ent_idx_col, ent_type_col = [], [], [], []
        for r in dataframe_iter_rows(ent_df):
            ent_ids_col.append(r['entity_id'])
            ent_layer_col.append(r['layer_id'])
            ent_idx_col.append(r['idx'])
            ent_type_col.append(r.get('type', 'node'))

    # The canonical store is filled from these, and so is everything derived
    # from it, so the load describes each entity once.
    entity_definitions: list = []
    layer_cache: dict = {}  # layer_id -> tuple
    # Fallback for an entirely absent layer id: use the graph's current
    # placeholder rank (which may be ('_',) for a flat graph or
    # ('_','_',...) for a multi-aspect one).
    aspects = getattr(graph, '_aspects', ('_',))
    placeholder = tuple('_' for _ in aspects) if aspects != ('_',) else ('_',)

    for vid, lid, idx, kind in zip(
        ent_ids_col,
        ent_layer_col,
        ent_idx_col,
        ent_type_col or [None] * len(ent_ids_col),
        strict=False,
    ):
        layer_tuple = layer_cache.get(lid)
        if layer_tuple is None:
            stored = layer_dict.get_layer(lid)
            # Preserve the EXACT stored layer tuple — including legacy
            # ('_',) coords saved before aspects were declared, even if
            # the current graph has more aspects. Literal roundtrip wins
            # over rank normalisation.
            layer_tuple = tuple(stored) if stored is not None else placeholder
            layer_cache[lid] = layer_tuple
        stored_kind = kind or 'node'
        entity_definitions.append(
            (
                int(idx),
                _structure.EntityRef(
                    vid, LOADED_ENTITY_KIND.get(stored_kind, _structure.NODE), layer_tuple
                ),
            )
        )
    # A row is the address a matrix gives an entity, so the store numbers its
    # slots in the order the file records rather than in the order it lists them.
    entity_definitions.sort(key=lambda item: item[0])
    entity_definitions = [ref for _row, ref in entity_definitions]

    # 3. Edges — merged metadata + endpoint table
    edge_definitions: dict = {}
    edge_columns: list = []

    def _side(endpoint) -> frozenset:
        """One side of an edge, as the set of endpoints it names."""
        return frozenset() if endpoint is None else frozenset({endpoint})

    def _reassemble_endpoint(vid, layer_id):
        if vid is None:
            return None
        if layer_id is None:
            return vid
        layer_tuple = layer_dict.get_layer(int(layer_id))
        return (vid, layer_tuple) if layer_tuple is not None else vid

    ec = _columns_as_lists(
        dataframe_read_parquet(path / 'edges.parquet'),
        (
            'edge_id',
            'col_idx',
            'weight',
            'directed',
            'kind',
            'ml_kind',
            'source',
            'source_layer_id',
            'target',
            'target_layer_id',
            'hyper_layer_id',
        ),
    )
    for (
        eid,
        col_idx_raw,
        w,
        directed,
        kind_raw,
        ml_kind,
        src_vid,
        src_lid,
        tgt_vid,
        tgt_lid,
        hyper_lid,
    ) in zip(
        ec['edge_id'],
        ec['col_idx'],
        ec['weight'],
        ec['directed'],
        ec['kind'],
        ec['ml_kind'],
        ec['source'],
        ec['source_layer_id'],
        ec['target'],
        ec['target_layer_id'],
        ec['hyper_layer_id'],
        strict=False,
    ):
        col_idx = int(col_idx_raw)
        weight = float(w) if w is not None else 1.0
        kind = kind_raw or 'binary'

        if kind == 'hyper':
            hyper_ml_layers = (
                layer_dict.get_layer(int(hyper_lid)) if hyper_lid is not None else None
            )
            # Endpoints come from hyperedge_definitions.parquet; populate later.
            definition_layers = hyper_ml_layers
            source, target = frozenset(), frozenset()
        else:
            src = _reassemble_endpoint(src_vid, src_lid)
            tgt = _reassemble_endpoint(tgt_vid, tgt_lid)
            # Reconstruct ml_layers from the endpoint coords for multilayer
            # binary edges so layer_edge_set/_effective_ml_edge_kind keep
            # working after a roundtrip.
            if (
                isinstance(src, tuple)
                and isinstance(tgt, tuple)
                and len(src) == 2
                and len(tgt) == 2
                and isinstance(src[1], tuple)
                and isinstance(tgt[1], tuple)
            ):
                binary_ml_layers = (src[1], tgt[1])
            else:
                binary_ml_layers = None
            definition_layers = binary_ml_layers
            source, target = _side(src), _side(tgt)

        edge_columns.append((col_idx, eid))
        edge_definitions[eid] = _structure.EdgeDefinition(
            id=eid,
            kind=LOADED_EDGE_KIND.get(kind, _structure.BINARY),
            source=source,
            target=target,
            weight=weight,
            directed=directed,
            ml_kind=ml_kind,
            ml_layers=definition_layers,
        )

    # 4. Hyperedge member / head / tail
    hyper_path = path / 'hyperedge_definitions.parquet'
    if hyper_path.exists():
        for hrow in dataframe_iter_rows(dataframe_read_parquet(hyper_path)):
            eid = hrow['edge_id']
            if eid not in edge_definitions:
                continue
            is_dir = bool(hrow.get('directed', False))
            if is_dir:
                source = frozenset(_deserialize_hyper_members(hrow.get('head')))
                target = frozenset(_deserialize_hyper_members(hrow.get('tail')))
            else:
                source = frozenset(_deserialize_hyper_members(hrow.get('members')))
                target = frozenset()
            edge_definitions[eid] = edge_definitions[eid]._replace(
                source=source, target=target, directed=is_dir
            )

    # 5. Explicit coefficients — records data, restored exactly. Files written before
    # edge_coeffs.parquet existed kept coefficients only inside the matrix, so fall
    # back to recovering them from it when the table is absent.
    coeff_path = path / 'edge_coeffs.parquet'
    if coeff_path.exists():
        cc = _columns_as_lists(
            dataframe_read_parquet(coeff_path),
            ('edge_id', 'entity_id', 'layer_id', 'coeff'),
        )
        for eid, vid, lid, val in zip(
            cc['edge_id'], cc['entity_id'], cc['layer_id'], cc['coeff'], strict=False
        ):
            definition = edge_definitions.get(eid)
            if definition is None:
                continue
            if definition.coefficients is None:
                definition = definition._replace(coefficients={})
                edge_definitions[eid] = definition
            definition.coefficients[_reassemble_endpoint(vid, lid)] = float(val)

    # 5b. The flexible-direction policy of each edge that declares one. A file
    # written before the format carried it has no such table, which reads as no
    # edge having one.
    policy_path = path / 'edge_direction_policy.json'
    if policy_path.exists():
        import json as _json

        for eid, policy in _json.loads(policy_path.read_text()).items():
            definition = edge_definitions.get(eid)
            if definition is not None:
                edge_definitions[eid] = definition._replace(direction_policy=policy)

    # 6. Install the whole structure at once. This fills the canonical store of
    # the graph from the definitions alone, so a graph read from a file answers
    # exactly as one built by hand.
    # A column is the address a matrix gives an edge, so the store takes the
    # edges in the order the file records rather than in the order it lists them.
    edge_columns.sort(key=lambda item: item[0])
    graph._install_structure(
        definitions=(
            entity_definitions,
            [edge_definitions[eid] for _column, eid in edge_columns],
        ),
    )

    # 7. Coefficients a legacy file kept only inside its matrix. This needs the
    # installed structure, because a column names its members by row.
    if not coeff_path.exists() and matrix is not None:
        _recover_legacy_coeffs(graph, matrix)


def _recover_legacy_coeffs(graph, csc) -> None:
    """Recover the coefficients a legacy file kept only inside its matrix.

    A file written before the coefficient table existed records an explicit
    coefficient nowhere else. So a column the edge cannot derive from its weight
    and its directedness carries one, and the edge takes the column as it stands.

    The matrix is the one the file holds, not the one the graph derives. The
    derived one says exactly what the store says, so reading it here would
    recover nothing.
    """
    n_cols = csc.shape[1]
    for edge in _structure.iter_edges(graph, include_placeholders=True):
        column = _structure.edge_column(graph, edge.id)
        if column < 0 or column >= n_cols:
            continue
        start, end = int(csc.indptr[column]), int(csc.indptr[column + 1])
        if start == end:
            continue
        stored = {}
        for row, value in zip(csc.indices[start:end], csc.data[start:end], strict=False):
            try:
                key = _structure.entity_key_of_row(graph, int(row))
            except KeyError:
                continue
            if float(value) != 0.0:
                stored[key] = float(value)
        if not stored:
            continue
        # Compare in float32. The matrix holds float32 while a weight is a Python
        # float, so a float64 comparison never matches for an ordinary weight such
        # as 0.1, and every column would masquerade as stoichiometry.
        derived = {
            key: float(np.float32(value))
            for key, value in _structure.edge_members(graph, edge.id).items()
            if value != 0.0
        }
        if stored != derived:
            graph._replace_edge_coeffs(edge.id, stored)


# The package said "vertex" where it now says "node", so an archive written
# before the rename holds the old word in four of its member names and in two of
# its columns. A reader takes both spellings and the writer emits the new ones
# alone, which is the only direction that has to hold.
_RENAMED_MEMBERS = {
    '_node_table.parquet': '_vertex_table.parquet',
    'node_presence.parquet': 'vertex_presence.parquet',
    'node_memberships.parquet': 'vertex_memberships.parquet',
    'node_layer_attributes.parquet': 'vertex_layer_attributes.parquet',
}
_RENAMED_COLUMNS = {'vertex_id': 'node_id', 'vertex_layer_id': 'node_layer_id'}


def _member(path: Path, name: str) -> Path | None:
    """Return the path of one archive member, under its name or the one it had."""
    member = path / name
    if member.exists():
        return member
    older = _RENAMED_MEMBERS.get(name)
    if older is None:
        return None
    older_member = path / older
    return older_member if older_member.exists() else None


def _read_member(path: Path, name: str):
    """Read one archive member, with the columns named the way the package names them."""
    member = _member(path, name)
    if member is None:
        return None
    table = dataframe_read_parquet(member)
    held = set(dataframe_columns(table))
    mapping = {old: new for old, new in _RENAMED_COLUMNS.items() if old in held and new not in held}
    return rename_dataframe_columns(table, mapping) if mapping else table


def _load_tables(graph, path: Path):
    """Load annotation tables with the configured dataframe backend."""
    graph._node_table = _read_member(path, '_node_table.parquet')
    graph._edge_table = dataframe_read_parquet(path / '_edge_table.parquet')
    graph.slice_attributes = dataframe_read_parquet(path / 'slice_attributes.parquet')
    graph.edge_slice_attributes = dataframe_read_parquet(path / 'edge_slice_attributes.parquet')


def _load_multilayers(graph, path: Path, layer_dict: _LayerDict):
    """Load Kivela multilayer structures from v2 layout (int layer ids)."""
    import json

    if not path.exists() or not (path / 'metadata.json').exists():
        return

    _UNSET = object()  # sentinel for cache miss
    legacy_flat_nodes = {
        ref.id
        for ref in _structure.iter_entities(graph)
        if ref.kind == _structure.NODE and ref.layer == ('_',)
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

    # Cache resolved layer_id -> [list] coercions; at UC1 scale layer IDs
    # repeat ~1.87M times across node_presence rows but only ~100 distinct
    # values, so caching turns a per-row tuple-to-list copy into a dict hit.
    _layer_list_cache: dict = {}

    def _layer_list(layer_id):
        cached = _layer_list_cache.get(layer_id, _UNSET)
        if cached is not _UNSET:
            return cached
        t = layer_dict.get_layer(layer_id) if layer_id is not None else None
        out = list(t) if t is not None else []
        _layer_list_cache[layer_id] = out
        return out

    vm_df = _read_member(path, 'node_presence.parquet')
    if vm_df is not None:
        # Read columns once and zip, avoiding the per-row dict allocation
        # that `dataframe_iter_rows` produces.
        try:
            import polars as _pl

            if isinstance(vm_df, _pl.DataFrame):
                vids = vm_df['node_id'].to_list()
                lids = vm_df['layer_id'].to_list()
            else:
                raise TypeError('not polars')
        except (ImportError, TypeError):
            vids, lids = [], []
            for r in dataframe_iter_rows(vm_df):
                vids.append(r['node_id'])
                lids.append(r['layer_id'])
        multilayer['VM'] = [
            {'node': vid, 'layer': _layer_list(lid)} for vid, lid in zip(vids, lids, strict=False)
        ]

    raw_edge_layers: dict = {}
    if (path / 'edge_layers.parquet').exists():
        el_df = dataframe_read_parquet(path / 'edge_layers.parquet')
        # Polars column-access fast path: avoid the per-row dict alloc that
        # dataframe_iter_rows produces.
        try:
            import polars as _pl

            if isinstance(el_df, _pl.DataFrame):
                el_eids = el_df['edge_id'].to_list()
                el_l1s = el_df['layer_1_id'].to_list()
                el_l2s = (
                    el_df['layer_2_id'].to_list()
                    if 'layer_2_id' in el_df.columns
                    else [None] * len(el_eids)
                )
            else:
                raise TypeError('not polars')
        except (ImportError, TypeError):
            el_eids, el_l1s, el_l2s = [], [], []
            for row in dataframe_iter_rows(el_df):
                el_eids.append(row['edge_id'])
                el_l1s.append(row['layer_1_id'])
                el_l2s.append(row.get('layer_2_id'))
        # Reuse the layer_list cache to avoid duplicate get_layer lookups
        # and produce tuples directly (skip the serialize/deserialize round
        # trip the manifest path used to do).
        _layer_tuple_cache: dict = {}
        for eid, l1, l2 in zip(el_eids, el_l1s, el_l2s, strict=False):
            t1 = _layer_tuple_cache.get(l1, _UNSET)
            if t1 is _UNSET:
                t1 = layer_dict.get_layer(l1) if l1 is not None else None
                _layer_tuple_cache[l1] = t1
            if l2 is None:
                raw_edge_layers[eid] = t1
            else:
                t2 = _layer_tuple_cache.get(l2, _UNSET)
                if t2 is _UNSET:
                    t2 = layer_dict.get_layer(l2)
                    _layer_tuple_cache[l2] = t2
                raw_edge_layers[eid] = (t1, t2)

    # ml_kind ("intra"/"inter"/"coupling") reached the store from edges.parquet
    # in _load_structure — rebuild the manifest map here so
    # restore_multilayer_manifest can hand it back to the graph state machine.
    multilayer['edge_kind'] = {
        edge.id: edge.ml_kind
        for edge in _structure.iter_edges(graph, include_placeholders=True)
        if edge.ml_kind is not None
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

    vla_df = _read_member(path, 'node_layer_attributes.parquet')
    if vla_df is not None:
        multilayer['node_layer_attrs'] = [
            {
                'node': row['node_id'],
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

    # Apply raw edge_layers directly (skip the serialize→deserialize
    # round-trip the manifest path used to do).
    if raw_edge_layers:
        graph.edge_layers.update(raw_edge_layers)

    # Native format roundtrips preserve the stored entity-index coordinates
    # exactly, even when multilayer metadata is declared afterwards.
    if legacy_flat_nodes and graph.aspects:
        placeholder = tuple('_' for _ in graph.aspects)

        def _stored_key(key):
            node_id, coord = key
            if node_id in legacy_flat_nodes and coord == placeholder:
                return (node_id, ('_',))
            return key

        graph._remap_entity_keys(
            {
                ref.key: (ref.id, ('_',))
                for ref in _structure.iter_entities(graph)
                if ref.id in legacy_flat_nodes and ref.layer == placeholder
            }
        )
        if graph._state_attrs:
            graph._state_attrs = {
                _stored_key(key): attrs for key, attrs in graph._state_attrs.items()
            }


def _load_slices(graph, path: Path, layer_dict: _LayerDict):
    """Reconstruct slice registry and memberships from v2 layout."""
    registry_df = dataframe_read_parquet(path / 'registry.parquet')
    existing_slices = set(graph.slices.list(include_default=True))
    # Column access — single .to_list() call avoids per-row dict alloc.
    try:
        import polars as _pl

        reg_is_polars = isinstance(registry_df, _pl.DataFrame)
    except ImportError:
        reg_is_polars = False
    if reg_is_polars:
        reg_ids = registry_df['slice_id'].to_list()
    else:
        reg_ids = [r['slice_id'] for r in dataframe_iter_rows(registry_df)]
    for slice_id in reg_ids:
        if slice_id not in existing_slices:
            graph.slices.add(slice_id)
            existing_slices.add(slice_id)

    # Node memberships (node_layer_id may be null for bare-vid memberships)
    node_df = _read_member(path, 'node_memberships.parquet')
    if reg_is_polars and isinstance(node_df, _pl.DataFrame):
        vm_slice = node_df['slice_id'].to_list()
        vm_bare = node_df['node_id'].to_list()
        vm_lid = (
            node_df['node_layer_id'].to_list()
            if 'node_layer_id' in node_df.columns
            else [None] * len(vm_bare)
        )
    else:
        vm_slice, vm_bare, vm_lid = [], [], []
        for row in dataframe_iter_rows(node_df):
            vm_slice.append(row['slice_id'])
            vm_bare.append(row['node_id'])
            vm_lid.append(row.get('node_layer_id'))
    # Cache layer lookups, and group the entities by bare id once instead of
    # asking the graph about each membership row.
    _vlayer_cache: dict = {}
    entities_by_id = _structure.entities_by_id(graph)
    for slice_id, bare, layer_id in zip(vm_slice, vm_bare, vm_lid, strict=False):
        if bare not in entities_by_id:
            continue
        if layer_id is None:
            node_id = bare
        else:
            layer_tuple = _vlayer_cache.get(layer_id)
            if layer_tuple is None:
                layer_tuple = layer_dict.get_layer(int(layer_id))
                _vlayer_cache[layer_id] = layer_tuple
            node_id = (bare, layer_tuple) if layer_tuple is not None else bare
        graph.slices.add_node_to_slice(slice_id, node_id)

    slice_membership: dict[str, list[str]] = {}
    slice_weights: dict[str, dict[str, float]] = {}
    edge_df = dataframe_read_parquet(path / 'edge_memberships.parquet')
    if reg_is_polars and isinstance(edge_df, _pl.DataFrame):
        em_slice = edge_df['slice_id'].to_list()
        em_eid = edge_df['edge_id'].to_list()
        em_w = edge_df['weight'].to_list() if 'weight' in edge_df.columns else [None] * len(em_eid)
        for lid, eid, w in zip(em_slice, em_eid, em_w, strict=False):
            slice_membership.setdefault(lid, []).append(eid)
            if w is not None:
                slice_weights.setdefault(lid, {})[eid] = w
    else:
        for row in dataframe_to_rows(edge_df):
            lid = row['slice_id']
            eid = row['edge_id']
            slice_membership.setdefault(lid, []).append(eid)
            w = row.get('weight', None)
            if w is not None:
                slice_weights.setdefault(lid, {})[eid] = w

    # `attach_edges` rather than `add_edges`, because the node memberships are
    # already loaded above and `add_edges` drops an edge that occupies no column.
    for lid, eids in slice_membership.items():
        graph.slices.attach_edges(lid, eids)

    # Per-slice edge weights (preserve overrides) — bulk-update via the
    # public attrs API so the slice_edge_weights cache and the
    # edge_slice_attributes parquet stay consistent.
    for lid, per_edge in slice_weights.items():
        if not per_edge:
            continue
        graph.attrs.set_edge_slice_attrs_bulk(
            lid,
            {eid: {'weight': float(w)} for eid, w in per_edge.items()},
        )


def _load_audit(graph, path: Path):
    """Load history and provenance."""
    import json as _json

    history_path = path / 'history.parquet'
    if history_path.exists():
        history_df = dataframe_read_parquet(history_path)
        graph._history = dataframe_to_rows(history_df)

    snapshots_path = path / 'snapshots.json'
    if snapshots_path.exists():
        graph._snapshots = [
            {
                'label': snap.get('label'),
                'version': snap.get('version'),
                'node_ids': set(snap.get('node_ids', []) or []),
                'edge_ids': set(snap.get('edge_ids', []) or []),
                'slice_ids': set(snap.get('slice_ids', []) or []),
            }
            for snap in _json.loads(snapshots_path.read_text())
        ]


def _load_uns(graph, path: Path):
    """Load unstructured metadata."""
    import json

    attrs_path = path / 'graph_attributes.json'
    if attrs_path.exists():
        graph.graph_attributes = json.loads(attrs_path.read_text())
