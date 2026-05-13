from .graph import AnnNet
from ._records import (
    EdgeType,
    EntityRecord,  # TODO: remove dependency on internal _records.py classes (exported for io/cx2.py)
)

Graph = AnnNet

__all__ = ['AnnNet', 'Graph', 'EdgeType', 'EntityRecord']
