"""Round-trip integrity test for the native .annnet format on multilayer graphs.

Covers a non-trivial graph shape: a named slice with bare-vid vertices and
binary edges (PKN-style base), a multi-elementary-layer aspect, per-layer
vertex inserts via ``G.add_vertices(..., layer=aa)``, intra-layer edges with
``(vid, layer_coord)`` tuple endpoints via ``G.add_edges([...])``, and per
``(vertex, layer)`` attributes.

Verifies that ``write`` + ``read`` is lossless for:
    - global vertex/edge counts (slice-membership domain)
    - explicit slice vertex and edge sets
    - layer-induced edge sets
    - per-layer vertex attributes
    - the aspects/elementary-layers structure
    - slice membership invariant: vertex sets contain only bare vid strings
      (no leaked ``(vid, layer_coord)`` tuples)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import annnet
from annnet.io import read as io_read
from annnet.io import write as io_write


def _build_multilayer_graph():
    G = annnet.AnnNet(directed=True)

    base_vertices = ['TP53', 'MDM2', 'EGFR', 'AKT1', 'PTEN', 'MYC', 'BRCA1', 'ATM']
    G.add_vertices(base_vertices, slice='base_pkn')
    G.add_edges(
        [
            {'source': 'EGFR', 'target': 'AKT1', 'weight': 1.0},
            {'source': 'AKT1', 'target': 'MDM2', 'weight': 0.8},
            {'source': 'MDM2', 'target': 'TP53', 'weight': -1.0},
            {'source': 'TP53', 'target': 'MYC', 'weight': 0.5},
            {'source': 'PTEN', 'target': 'AKT1', 'weight': -0.7},
            {'source': 'ATM', 'target': 'BRCA1', 'weight': 1.0},
            {'source': 'ATM', 'target': 'TP53', 'weight': 0.9},
        ],
        slice='base_pkn',
    )

    samples = ['S1', 'S2', 'S3']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.set_aspects(['sample'], {'sample': samples})

    intra_edge_specs = [
        ('EGFR', 'AKT1', 0.8),
        ('AKT1', 'MDM2', 0.7),
        ('TP53', 'MYC', 0.6),
    ]
    for s in samples:
        aa = (s,)
        G.add_vertices(base_vertices, layer=aa)
        for v in base_vertices:
            G.layers.set_vertex_layer_attrs(v, aa, expr=float(hash((v, s)) % 1000) / 100.0)
        G.add_edges(
            [
                {'source': (src, aa), 'target': (tgt, aa), 'weight': w}
                for src, tgt, w in intra_edge_specs
            ]
        )

    G.layers.add_elementary_layer('sample', 'consensus')
    consensus_aa = ('consensus',)
    consensus_nodes = ['TP53', 'AKT1', 'MYC']
    G.add_vertices(consensus_nodes, layer=consensus_aa)
    for v in consensus_nodes:
        G.layers.set_vertex_layer_attrs(v, consensus_aa, score=0.42, frequency=0.66)

    return G, samples, base_vertices, consensus_nodes, intra_edge_specs


def _snapshot(G, samples, probe_vid, consensus_aa):
    return {
        'nv': G.global_count('vertices'),
        'ne': G.global_count('edges'),
        'slice_base_v': set(G.slices.vertices('base_pkn')),
        'slice_base_e': set(G.slices.edges('base_pkn')),
        'intra_per_sample': {s: len(G.layers.layer_edge_set((s,))) for s in samples},
        'attr_probe': {s: G.layers.get_vertex_layer_attrs(probe_vid, (s,)) for s in samples},
        'consensus_attr': G.layers.get_vertex_layer_attrs(probe_vid, consensus_aa),
        'aspects': tuple(G.aspects),
    }


def _assert_no_tuple_in_slice_vertices(G):
    for sid in G.slices.list(include_default=True):
        for v in G.slices.vertices(sid):
            assert isinstance(v, str), (
                f'slice {sid!r} contains non-string vertex {v!r} '
                f'({type(v).__name__}); slice membership must track bare vids'
            )


def test_annnet_format_multilayer_roundtrip(tmp_path: Path):
    G, samples, base_vertices, consensus_nodes, _ = _build_multilayer_graph()

    _assert_no_tuple_in_slice_vertices(G)
    before = _snapshot(G, samples, probe_vid='TP53', consensus_aa=('consensus',))

    out = tmp_path / 'graph.annnet'
    io_write(G, out, overwrite=True)
    assert out.exists() and out.stat().st_size > 0

    G2 = io_read(out)
    _assert_no_tuple_in_slice_vertices(G2)
    after = _snapshot(G2, samples, probe_vid='TP53', consensus_aa=('consensus',))

    assert before == after, f'round-trip diverged:\n  before: {before}\n  after:  {after}'


def test_annnet_format_directory_mode_roundtrip(tmp_path: Path):
    G, samples, _, _, _ = _build_multilayer_graph()
    before = _snapshot(G, samples, probe_vid='TP53', consensus_aa=('consensus',))

    out = tmp_path / 'graph_dir.annnet'
    io_write(G, out, overwrite=True)
    assert out.is_dir() or out.is_file()

    G2 = io_read(out)
    after = _snapshot(G2, samples, probe_vid='TP53', consensus_aa=('consensus',))
    assert before == after


@pytest.mark.parametrize('compression', ['zstd'])
def test_multilayer_intra_edges_preserve_ml_kind(tmp_path: Path, compression):
    """Intra-layer edges added via tuple endpoints must remain intra after read."""
    G, samples, _, _, intra_edge_specs = _build_multilayer_graph()

    out = tmp_path / 'graph.annnet'
    io_write(G, out, overwrite=True, compression=compression)
    G2 = io_read(out)

    for s in samples:
        aa = (s,)
        edges = G2.layers.layer_edge_set(aa)
        assert len(edges) == len(intra_edge_specs), (
            f'sample {s}: expected {len(intra_edge_specs)} intra edges, got {len(edges)}'
        )


if __name__ == '__main__':
    import sys

    sys.exit(pytest.main([__file__, '-v']))
