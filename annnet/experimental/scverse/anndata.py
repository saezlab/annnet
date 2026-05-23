"""AnnData bridges for AnnNet."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core import AnnNet
from ._shared import (
    ANNNET_UNS_KEY,
    add_edges_from_var,
    obs_spatial_matrix,
    require_dependency,
    restore_multilayer,
    build_obs_dataframe,
    build_var_dataframe,
    add_vertices_from_obs,
    build_annnet_manifest,
    build_vertex_incidence,
    infer_directed_from_var,
    restore_attrs_from_manifest,
    restore_edge_attrs_from_var,
    restore_vertices_from_obs_attrs,
)


def to_anndata(
    graph: AnnNet,
    *,
    include_private: bool = False,
    spatial_columns: tuple[str, str] | None = None,
) -> Any:
    """Export an AnnNet graph as an AnnData object.

    `obs` stores vertex-entity rows, `var` stores structural edge rows, and
    `X` stores the vertex-by-edge incidence matrix. AnnNet-only concepts such
    as slices, multilayer state, and raw attribute tables are preserved in
    `uns["__annnet__"]`.
    """
    ad = require_dependency('anndata', 'annnet[scverse] or pip install anndata')

    obs = build_obs_dataframe(graph, include_private=include_private)
    var = build_var_dataframe(graph, include_private=include_private)
    uns = {ANNNET_UNS_KEY: build_annnet_manifest(graph)}

    kwargs: dict[str, Any] = {}
    spatial = obs_spatial_matrix(obs, spatial_columns=spatial_columns)
    if spatial is not None:
        key, value = spatial
        kwargs['obsm'] = {key: value}

    return ad.AnnData(
        X=build_vertex_incidence(graph),
        obs=obs,
        var=var,
        uns=uns,
        **kwargs,
    )


def from_anndata(
    adata: Any,
    *,
    annotations_backend: str | None = 'auto',
) -> AnnNet:
    """Restore an AnnNet graph from an AnnData object.

    When the input was produced by :func:`to_anndata`, restoration is lossless
    for the supported AnnNet structural surface. Generic AnnData objects are
    also accepted when `var` exposes edge structure via `source`/`target` or
    `members`/`head`/`tail`.
    """
    manifest = dict(getattr(adata, 'uns', {}).get(ANNNET_UNS_KEY, {}) or {})
    directed = manifest.get('directed', infer_directed_from_var(adata.var))
    graph = AnnNet(directed=directed, annotations_backend=annotations_backend)

    obs = adata.obs.copy()
    if not isinstance(obs, pd.DataFrame):
        obs = pd.DataFrame(obs)
    var = adata.var.copy()
    if not isinstance(var, pd.DataFrame):
        var = pd.DataFrame(var)

    add_vertices_from_obs(graph, obs)
    restore_vertices_from_obs_attrs(graph, obs)
    add_edges_from_var(graph, var)
    restore_edge_attrs_from_var(graph, var)

    restore_multilayer(graph, manifest.get('multilayer', {}) or {})
    if manifest:
        restore_attrs_from_manifest(graph, manifest)
    else:
        graph.uns.update(dict(getattr(adata, 'uns', {}) or {}))

    return graph
