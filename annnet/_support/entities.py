"""Edge entities across formats that have no notion of one.

An edge entity is an edge that is also an endpoint: another edge can name it.
Most exchange formats hold nodes and edges and nothing in between, so a writer
that meets one has three honest choices: write it in a place the format has for
it, record it beside the file, or say that it was left out. Reading it back as a
plain node is not among them, because the graph that comes out then answers
questions the graph that went in never would.

The record is built by :func:`annnet.core._structure.edge_entity_record`. This
module gives it back: :func:`restore_entities` puts the kinds on a graph a reader
has already built. The graph is used through its own doors only, so nothing here
depends on the core.
"""

from __future__ import annotations

from typing import Any
import warnings

from .loss import AnnNetLossWarning

__all__ = ['restore_entities', 'update_edge_attributes']


def restore_entities(graph, record: dict[str, Any] | None) -> None:
    """Give a rebuilt graph back the edge entities ``record`` describes.

    The graph is expected to hold the entities as nodes (or not at all) and the
    connecting edges as binary edges, which is what a reader that ignores edge
    entities produces. Attributes stored under an entity's identity that cannot be
    placed are reported as a loss and skipped; the rest is restored.
    """
    if not record:
        return
    keys = [(item['id'], tuple(item.get('layer') or ('_',))) for item in record.get('entities', ())]
    if keys:
        graph._declare_edge_entities(keys, record.get('node_edges', ()))
    rows = {
        entity_id: {name: value for name, value in row.items() if value is not None}
        for entity_id, row in (record.get('attributes') or {}).items()
    }
    update_edge_attributes(graph, {entity_id: row for entity_id, row in rows.items() if row})


def update_edge_attributes(graph, updates: dict[str, dict[str, Any]]) -> None:
    """Merge attribute rows into the edges of ``graph`` that exist.

    A record written before edge entities were recorded can carry attributes for
    an identity the reader rebuilt as a node. Those rows have no edge to go to, so
    they are reported as a loss instead of failing the read.
    """
    if not updates:
        return
    try:
        graph.attrs.update('edges', updates)
        return
    except KeyError:
        pass
    placed, skipped = {}, []
    for key, row in updates.items():
        try:
            graph.attrs.row('edges', key)
        except KeyError:
            skipped.append(str(key))
        else:
            placed[key] = row
    if placed:
        graph.attrs.update('edges', placed)
    if skipped:
        warnings.warn(
            f'attributes for identities that are not edges were left out: {sorted(skipped)!r}',
            AnnNetLossWarning,
            stacklevel=2,
        )
