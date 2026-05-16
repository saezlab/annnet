"""scverse interoperability bridges for AnnNet."""

from __future__ import annotations

from .mudata import to_mudata, from_mudata
from .anndata import to_anndata, from_anndata
from .spatialdata import to_spatialdata, from_spatialdata

__all__ = [
    'from_anndata',
    'from_mudata',
    'from_spatialdata',
    'to_anndata',
    'to_mudata',
    'to_spatialdata',
]
