"""The words a stored file and an older API use for the kinds of the core.

The facade uses the vocabulary of the new core. A file written before the refactor,
and a table a user already reads, use the words the record store used. This map is
the one place the two meet, so a writer maps back and a reader maps forward.
"""

from __future__ import annotations

from . import _structure as S

STORED_ENTITY_KIND = {
    S.NODE: 'node',
    S.EDGE_ENTITY: 'edge_entity',
}

STORED_EDGE_KIND = {
    S.BINARY: 'binary',
    S.HYPER: 'hyper',
    S.NODE_EDGE: 'node_edge',
    S.PLACEHOLDER: 'edge_placeholder',
}
