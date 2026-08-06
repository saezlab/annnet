"""The core of the package: one graph object and the records it hands back.

What this module exports is the whole of what code outside ``annnet.core`` may
name. Every other module here is private: the canonical store, the mutation
gateway, the query facade, the attribute columns, the matrix builders and the
namespaces are all reachable through the graph object and through nothing else.

Input-output code, adapters and bridges read structure through the query facade
``annnet.core._structure`` and never through the store behind it (FR-002).
"""

from .graph import AnnNet
from ._records import EdgeType, EdgeView, NodeView

Graph = AnnNet

__all__ = [
    'AnnNet',
    'Graph',
    'EdgeType',
    'EdgeView',
    'NodeView',
]
