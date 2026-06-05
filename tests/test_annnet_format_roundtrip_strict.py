"""Strict literal roundtrip for the native .annnet format.

This builds a fixture graph that exercises every observable the format is
supposed to preserve (multilayer aspects + elementary layers, intra/inter/
coupling/hyper edges, slice membership + per-slice weights + slice attrs,
vertex/edge/layer attribute tables, history snapshots, composite-vertex-key
indexing, edge direction policies, graph-level uns metadata), then runs
build → write → read → write → read and asserts equality on every
state-bearing field.

The test deliberately reaches into private state (`_entities`, `_edges`,
matrix coordinates, slice records). It is intended to fail loudly the
moment any field stops surviving the roundtrip.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import pytest

from annnet import AnnNet
from annnet.io import read, write


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _build_fixture() -> AnnNet:
    G = AnnNet(directed=True)

    # Two aspects so we exercise composite layer coords.
    G.layers.set_aspects(
        ['level', 'phase'], {'level': ['sig', 'cpx', 'reg'], 'phase': ['t0', 't1']}
    )

    # Aspect-level attrs.
    G.layers.set_aspect_attrs('level', description='biological process')
    G.layers.set_aspect_attrs('phase', description='time bin')

    # Elementary-layer attrs (per aspect/value).
    G.layers.set_elementary_layer_attrs('level', 'sig', color='#1f77b4', weight=1.0)
    G.layers.set_elementary_layer_attrs('phase', 't0', when='baseline')

    # Per layer-tuple attrs.
    G.layers.set_layer_attrs(('sig', 't0'), note='intra-signaling at baseline')

    # Vertices: protein has presence in multiple layer combinations.
    G.add_vertices(
        [
            {'vertex_id': 'prot:A', 'kind': 'protein', 'sym': 'A', 'score': 0.1},
            {'vertex_id': 'prot:B', 'kind': 'protein', 'sym': 'B', 'score': 0.4},
            {'vertex_id': 'prot:C', 'kind': 'protein', 'sym': 'C', 'score': 0.7},
        ],
        layer=('sig', 't0'),
    )
    G.add_vertices(
        [
            {'vertex_id': 'prot:A', 'kind': 'protein', 'sym': 'A', 'score': 0.1},
            {'vertex_id': 'prot:B', 'kind': 'protein', 'sym': 'B', 'score': 0.4},
        ],
        layer=('cpx', 't0'),
    )
    G.add_vertices(
        [{'vertex_id': 'gene:A', 'kind': 'gene', 'sym': 'A'}],
        layer=('reg', 't0'),
    )
    G.add_vertices(
        [{'vertex_id': 'prot:A', 'kind': 'protein', 'sym': 'A', 'score': 0.2}],
        layer=('sig', 't1'),
    )

    # Per-(vertex, layer) attrs (state_attrs).
    G.layers.set_vertex_layer_attrs('prot:A', ('sig', 't0'), abundance=12.5)
    G.layers.set_vertex_layer_attrs('prot:A', ('sig', 't1'), abundance=18.0)

    # Edges — cover every kind.
    sig0 = ('sig', 't0')
    sig1 = ('sig', 't1')
    cpx0 = ('cpx', 't0')
    reg0 = ('reg', 't0')

    # intra-layer binary
    G.add_edges(
        [
            {
                'source': ('prot:A', sig0),
                'target': ('prot:B', sig0),
                'edge_kind': 'signaling',
                'weight': 0.5,
                'edge_id': 'e_sig_AB',
            },
            {
                'source': ('prot:B', sig0),
                'target': ('prot:C', sig0),
                'edge_kind': 'signaling',
                'weight': -1.0,
                'edge_id': 'e_sig_BC',
            },
        ],
        default_edge_directed=True,
    )

    # inter-layer binary (same level, different phase)
    G.add_edges(
        [
            {
                'source': ('prot:A', sig0),
                'target': ('prot:A', sig1),
                'edge_kind': 'temporal_self',
                'edge_id': 'e_temp_A',
            }
        ],
        default_edge_directed=True,
    )

    # coupling binary (same vid, different aspect value on `level`)
    G.add_edges(
        [
            {
                'source': ('prot:A', sig0),
                'target': ('prot:A', cpx0),
                'edge_kind': 'identity',
                'edge_directed': False,
                'edge_id': 'e_id_A',
            }
        ]
    )

    # gene→prot translation (different vid, different layer)
    G.add_edges(
        [
            {
                'source': ('gene:A', reg0),
                'target': ('prot:A', sig0),
                'edge_kind': 'translation',
                'edge_id': 'e_trans_A',
            }
        ],
        default_edge_directed=True,
    )

    # hyperedge in complex layer (bare-vid members + layer kwarg)
    G.add_edges(
        [
            {
                'members': ['prot:A', 'prot:B'],
                'edge_id': 'cpx:AB',
                'edge_kind': 'complex',
                'weight': 1.0,
            }
        ],
        layer=cpx0,
    )

    # directed hyperedge in signaling layer (head/tail)
    G.add_edges(
        [
            {
                'head': ['prot:C'],
                'tail': ['prot:A', 'prot:B'],
                'edge_id': 'rxn:1',
                'edge_kind': 'rxn',
                'edge_directed': True,
            }
        ],
        layer=sig0,
    )

    # Slices (overlay; orthogonal to layers).
    G.slices.add('cytosol', source='HPA')
    G.slices.add_vertex_to_slice('cytosol', 'prot:A')
    G.slices.add_vertex_to_slice('cytosol', 'prot:B')
    G.slices.add_edges('cytosol', ['e_sig_AB'])
    G.slices.add('nucleus', source='HPA')
    G.slices.add_vertex_to_slice('nucleus', 'gene:A')

    # Per-slice edge weight override.
    G.attrs.set_edge_slice_attrs('cytosol', 'e_sig_AB', weight=0.75)

    # Per-slice attribute (slice metadata table).
    G.attrs.set_slice_attrs('cytosol', notes='cytosolic subset')

    # Edge-entity (null endpoints) — sometimes used as a reified anchor.
    G.add_edges(
        [{'edge_id': 'entity:reified_1', 'edge_kind': 'placeholder'}],
        as_entity=True,
    )

    # uns / graph-level metadata.
    G.uns['cell_line'] = 'HEK293'
    G.uns['threshold'] = 0.05
    G.uns['authors'] = ['alice', 'bob']

    # History — take a couple of snapshots so the trail is non-trivial.
    G.history.snapshot('mid')
    G.history.snapshot('final')

    return G


# ---------------------------------------------------------------------------
# Snapshot of every observable
# ---------------------------------------------------------------------------


def _normalise_frozenset(value):
    if isinstance(value, frozenset):
        return tuple(sorted(value, key=repr))
    return value


def _edge_state(g: AnnNet):
    out = {}
    for eid, rec in g._edges.items():
        out[eid] = (
            _normalise_frozenset(rec.src),
            _normalise_frozenset(rec.tgt),
            float(rec.weight) if rec.weight is not None else None,
            rec.directed,
            rec.etype,
            rec.col_idx,
            rec.ml_kind,
            rec.ml_layers,
            rec.direction_policy,
        )
    return out


def _matrix_state(g: AnnNet):
    coo = g._matrix.tocoo()
    triples = sorted(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), strict=False))
    return triples, tuple(coo.shape), str(coo.dtype)


def _df_state(df):
    if df is None:
        return None
    if hasattr(df, 'to_dicts'):
        rows = df.to_dicts()
    elif hasattr(df, 'to_dict'):
        rows = df.to_dict(orient='records')
    else:
        return repr(df)
    normalised = []
    for r in rows:
        items = sorted(((k, v) for k, v in r.items()), key=lambda kv: kv[0])
        normalised.append(tuple((k, v) for k, v in items))
    return sorted(normalised, key=repr)


def _slice_state(g: AnnNet):
    out = {}
    for sid in g.slices.list(include_default=True):
        rec = g._slices[sid]
        out[sid] = {
            'vertices': sorted(rec['vertices']),
            'edges': sorted(rec['edges']),
            'edge_weights': dict(g.slice_edge_weights.get(sid, {})),
        }
    return out


def _full_snapshot(g: AnnNet):
    """Capture every observable we expect a roundtrip to preserve."""
    return {
        'directed': g.directed,
        'aspects': g.layers.list_aspects(),
        'layers_per_aspect': {
            a: sorted(g.layers.list_layers(a)) for a in g.layers.list_aspects() or ()
        },
        'aspect_attrs': {a: g.layers.get_aspect_attrs(a) for a in g.layers.list_aspects() or ()},
        'entities': sorted(g._entities.keys()),
        'entity_records': {ekey: (rec.row_idx, rec.kind) for ekey, rec in g._entities.items()},
        'row_to_entity': dict(g._row_to_entity),
        'vid_to_ekeys': {vid: sorted(keys, key=repr) for vid, keys in g._vid_to_ekeys.items()},
        'edges': _edge_state(g),
        'col_to_edge': dict(g._col_to_edge),
        'matrix': _matrix_state(g),
        'vertex_attributes': _df_state(g.vertex_attributes),
        'edge_attributes': _df_state(g.edge_attributes),
        'slice_attributes': _df_state(g.slice_attributes),
        'edge_slice_attributes': _df_state(g.edge_slice_attributes),
        'layer_attributes': _df_state(g.layer_attributes),
        'slices': _slice_state(g),
        'default_slice': g._default_slice,
        'current_slice': g._current_slice,
        'uns': dict(g.uns),
        'history_labels': [
            (s if isinstance(s, str) else s.get('label')) for s in g.history.list_snapshots()
        ],
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _diff(a, b, path=''):
    """Return a list of human-readable diffs between two nested structures."""
    diffs = []
    if type(a) is not type(b):
        diffs.append(f'{path}: type differs ({type(a).__name__} vs {type(b).__name__})')
        return diffs
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b), key=repr)
        for k in keys:
            if k not in a:
                diffs.append(f'{path}[{k!r}]: missing on left')
            elif k not in b:
                diffs.append(f'{path}[{k!r}]: missing on right')
            else:
                diffs.extend(_diff(a[k], b[k], f'{path}[{k!r}]'))
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f'{path}: length differs ({len(a)} vs {len(b)})')
            return diffs
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            diffs.extend(_diff(x, y, f'{path}[{i}]'))
        return diffs
    if a != b:
        diffs.append(f'{path}: {a!r} != {b!r}')
    return diffs


def test_annnet_literal_roundtrip_strict():
    """Two-cycle write/read on a fully-loaded fixture must preserve every observable."""
    G = _build_fixture()
    original = _full_snapshot(G)

    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as tmp:
        warnings.simplefilter('error', UserWarning)  # any placeholder warning fails the test
        p1 = os.path.join(tmp, 'first.annnet')
        p2 = os.path.join(tmp, 'second.annnet')
        write(G, p1)
        G2 = read(p1)
        after_first = _full_snapshot(G2)
        write(G2, p2)
        G3 = read(p2)
        after_second = _full_snapshot(G3)

    d1 = _diff(original, after_first)
    d2 = _diff(after_first, after_second)

    msg = []
    if d1:
        msg.append('build -> read diffs:\n  ' + '\n  '.join(d1[:30]))
    if d2:
        msg.append('read -> write -> read diffs:\n  ' + '\n  '.join(d2[:30]))
    assert not msg, '\n\n'.join(msg)


if __name__ == '__main__':
    # Allow running this file directly for quick iteration.

    pytest.main([__file__, '-vv', '-s'])
