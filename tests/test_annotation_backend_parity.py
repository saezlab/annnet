"""Annotation backends must produce equivalent obs/var.

For every supported backend (polars, pandas, pyarrow) the call:

    G.add_vertices(["A","B","C"])

must produce ``obs.shape[0] == G.nv == 3`` rows. Mixing single-form and
batch-form calls must not crash on schema concat.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


BACKENDS = ['polars', 'pandas', 'pyarrow']


def _obs_nrows(graph: AnnNet) -> int:
    obs = graph.obs
    if hasattr(obs, 'shape'):
        return int(obs.shape[0])
    return len(obs)


@pytest.mark.parametrize('backend', BACKENDS)
def test_add_vertices_list_no_attrs_populates_obs(backend):
    G = AnnNet(directed=False, annotations_backend=backend)
    G.add_vertices(['A', 'B', 'C'])
    assert G.nv == 3
    assert _obs_nrows(G) == 3


@pytest.mark.parametrize('backend', BACKENDS)
def test_add_vertices_list_with_attrs_populates_obs(backend):
    G = AnnNet(directed=False, annotations_backend=backend)
    G.add_vertices(['A', 'B', 'C'], kind='gene')
    assert G.nv == 3
    assert _obs_nrows(G) == 3


@pytest.mark.parametrize('backend', BACKENDS)
def test_mixed_single_and_batch_does_not_crash(backend):
    """Calling single-form (with attrs) then batch-form must not crash on
    column-width mismatch in the underlying dataframe concat."""
    G = AnnNet(directed=False, annotations_backend=backend)
    G.add_vertices('A', kind='gene')
    G.add_vertices(['B', 'C'])  # no attrs — width-1 internal frame
    assert G.nv == 3
    assert _obs_nrows(G) == 3


@pytest.mark.parametrize('backend', BACKENDS)
def test_mixed_attrs_and_no_attrs_keeps_all_rows(backend):
    """Adding some vertices with attrs and some without must yield obs with
    every vertex represented; missing attr cells become null/None."""
    G = AnnNet(directed=False, annotations_backend=backend)
    G.add_vertices(['A'], kind='gene')
    G.add_vertices(['B'])
    G.add_vertices(['C'], expression=2.0)
    assert G.nv == 3
    assert _obs_nrows(G) == 3
