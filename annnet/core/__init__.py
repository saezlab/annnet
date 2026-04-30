from .graph import AnnNet, EdgeType
from ._records import (
    EntityRecord,  # TODO: remove dependency on internal _records.py classes (exported for io/cx2.py)
)

Graph = AnnNet

__all__ = ['AnnNet', 'Graph', 'EdgeType', 'EntityRecord']
