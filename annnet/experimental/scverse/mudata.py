"""MuData bridges for AnnNet."""

from __future__ import annotations

from typing import Any
import warnings

from ._shared import ANNNET_UNS_KEY, require_dependency
from .anndata import to_anndata, from_anndata


def to_mudata(
    graph,
    *,
    modality: str = 'graph',
    include_private: bool = False,
    spatial_columns: tuple[str, str] | None = None,
) -> Any:
    """Wrap an AnnNet graph in a single-modality MuData container."""
    mudata = require_dependency('mudata', 'annnet[scverse] or pip install mudata')
    adata = to_anndata(
        graph,
        include_private=include_private,
        spatial_columns=spatial_columns,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module=r'mudata\..*')
        mdata = mudata.MuData({modality: adata})
    mdata.uns[ANNNET_UNS_KEY] = {'encoding': 'annnet-mudata', 'version': 1, 'modality': modality}
    return mdata


def from_mudata(
    mdata: Any,
    *,
    modality: str | None = None,
    annotations_backend: str | None = 'auto',
):
    """Restore an AnnNet graph from a MuData modality."""
    mod = getattr(mdata, 'mod', None)
    if mod is None:
        raise TypeError('Expected a MuData object with a .mod mapping of AnnData modalities.')

    if modality is None:
        manifest = dict(getattr(mdata, 'uns', {}).get(ANNNET_UNS_KEY, {}) or {})
        modality = manifest.get('modality')
    if modality is None:
        if len(mod) == 1:
            modality = next(iter(mod.keys()))
        elif 'graph' in mod:
            modality = 'graph'
        else:
            raise ValueError(
                'MuData contains multiple modalities; pass modality= to choose which one stores AnnNet.'
            )

    return from_anndata(mod[modality], annotations_backend=annotations_backend)
