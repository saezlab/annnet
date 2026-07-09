from .graph import AnnNet
from ._records import (
    EdgeType,
    EntityRecord,  # exported for io/cx2.py compatibility
)

Graph = AnnNet

__all__ = ['AnnNet', 'Graph', 'EdgeType', 'EntityRecord']
