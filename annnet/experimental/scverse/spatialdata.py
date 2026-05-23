"""SpatialData bridges for AnnNet."""

from __future__ import annotations

import copy
from typing import Any

from ._shared import ANNNET_UNS_KEY, require_dependency
from .anndata import to_anndata, from_anndata

SPATIALDATA_UNS_KEY = 'annnet'


def to_spatialdata(
    graph,
    *,
    table_name: str = 'graph',
    include_private: bool = False,
    spatial_columns: tuple[str, str] | None = None,
) -> Any:
    """Wrap an AnnNet graph as a table-first SpatialData object.

    This bridge is conservative by design: it stores the canonical AnnData
    payload as a SpatialData table without inventing fake spatial elements.
    If vertex coordinates are present in `obs`, they are exported via
    `AnnData.obsm["spatial"]`.
    """
    spatialdata = require_dependency(
        'spatialdata',
        'annnet[scverse] or pip install spatialdata',
    )
    adata = to_anndata(
        graph,
        include_private=include_private,
        spatial_columns=spatial_columns,
    )
    if ANNNET_UNS_KEY in adata.uns:
        adata.uns[SPATIALDATA_UNS_KEY] = copy.deepcopy(adata.uns[ANNNET_UNS_KEY])
        del adata.uns[ANNNET_UNS_KEY]

    table = adata
    try:
        from spatialdata.models import TableModel  # type: ignore
    except ImportError:
        TableModel = None

    if TableModel is not None:
        try:
            table = TableModel.parse(adata)
        except Exception:  # noqa: BLE001
            table = adata

    attrs = {ANNNET_UNS_KEY: {'encoding': 'annnet-spatialdata', 'version': 1, 'table': table_name}}
    return spatialdata.SpatialData(tables={table_name: table}, attrs=attrs)


def from_spatialdata(
    sdata: Any,
    *,
    table_name: str | None = None,
    annotations_backend: str | None = 'auto',
):
    """Restore an AnnNet graph from a SpatialData table."""
    tables = getattr(sdata, 'tables', None)
    if tables is None:
        raise TypeError('Expected a SpatialData object with a .tables mapping of AnnData tables.')

    if table_name is None:
        attrs = getattr(sdata, 'attrs', {}) or {}
        manifest = dict(attrs.get(ANNNET_UNS_KEY, {}) or {})
        table_name = manifest.get('table')
    if table_name is None:
        if len(tables) == 1:
            table_name = next(iter(tables.keys()))
        elif 'graph' in tables:
            table_name = 'graph'
        else:
            raise ValueError(
                'SpatialData contains multiple tables; pass table_name= to choose the AnnNet payload.'
            )

    adata = tables[table_name].copy()
    if SPATIALDATA_UNS_KEY in adata.uns and ANNNET_UNS_KEY not in adata.uns:
        adata.uns[ANNNET_UNS_KEY] = copy.deepcopy(adata.uns[SPATIALDATA_UNS_KEY])
    return from_anndata(adata, annotations_backend=annotations_backend)
