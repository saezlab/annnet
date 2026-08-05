"""The single graph-construction path.

Every code path that builds a fully-formed graph (copy, subgraph / flat-selection,
flatten-to-flat, IO load) installs its canonical state through here. This module
and ``_mutate`` are the only two that write the canonical store.
"""

from __future__ import annotations

from . import _store as ST, _derive as D, _structure as S
from ._records import SliceRecord

_SLOT_ENTITY_KIND = {S.NODE: ST.NODE, S.EDGE_ENTITY: ST.EDGE_ENTITY}
_SLOT_EDGE_KIND = {
    S.BINARY: ST.BINARY,
    S.HYPER: ST.HYPER,
    S.NODE_EDGE: ST.NODE_EDGE,
    S.PLACEHOLDER: ST.PLACEHOLDER,
}

# ---------------------------------------------------------------------------
# Slice registries
# ---------------------------------------------------------------------------


def clone_slices(src_slices, *, drop_attributes=False) -> dict:
    """Clone a slice registry (membership sets copied; attributes optional)."""
    out = {}
    for sid, meta in src_slices.items():
        out[sid] = SliceRecord(
            vertices=set(meta['vertices']),
            edges=set(meta['edges']),
            attributes={} if drop_attributes else dict(meta.get('attributes', {})),
        )
    return out


def slices_from_specs(specs) -> dict:
    """Build a slice registry from ``{sid: {'vertices','edges','attributes'}}`` specs."""
    return {
        sid: SliceRecord(
            vertices=set(spec.get('vertices', ())),
            edges=set(spec.get('edges', ())),
            attributes=dict(spec.get('attributes', {})),
        )
        for sid, spec in specs.items()
    }


# ---------------------------------------------------------------------------
# Structural install (the only place outside _mutate that assigns SoT fields)
# ---------------------------------------------------------------------------


def store_from_definitions(g, entities, edges):
    """Build the slot store of ``g`` from what a loader parsed.

    ``entities`` names every entity as an :class:`_structure.EntityRef`, in row
    order, and ``edges`` names every edge as an
    :class:`_structure.EdgeDefinition`, in column order. Both are the vocabulary
    the facade speaks, so a loader outside the core describes a graph without
    reaching for the store or for a record.

    A loader describes the graph it read and this fills the store from the
    description alone, which is why nothing outside the core has to know how a
    store holds a graph.
    """
    default = g.directed
    store = ST.CoreState(directed=default, aspects=g._aspects)
    for ref in entities:
        store.add_entity(ref.key, _SLOT_ENTITY_KIND.get(ref.kind, ST.NODE))
    for edge in edges:
        directed = S.resolved_direction(default, edge.kind, edge.directed, bool(edge.target))
        store.add_edge(
            edge.id,
            ST.members_from_endpoints(
                store,
                None,
                edge.source,
                edge.target,
                edge.coefficients,
                edge.weight,
                directed,
            ),
            kind=_SLOT_EDGE_KIND.get(edge.kind, ST.BINARY),
            directed=edge.directed,
            weight=edge.weight,
            explicit_coefficients=edge.coefficients is not None,
            ml_kind=edge.ml_kind,
            ml_layers=edge.ml_layers,
            direction_policy=edge.direction_policy,
        )
    return store


def rebuild_store(g):
    """Return the store a graph describes, built from scratch in one pass.

    The graph is read back as definitions through the query facade and the
    result is filled from those alone, so nothing of the store it came from is
    carried over. What it checks is that the store still says what its own
    definitions say: a write that changed one field and left a dependent one
    behind gives a different store here.

    This is what took over from the rebuild-from-records the store was checked
    against while there were two stores. It is a round trip rather than a second
    opinion, and ``_validate`` holds the rules that no round trip can check.
    """
    return store_from_definitions(g, *S.definitions_of(g))


def install_structure(g, *, store=None, definitions=None) -> None:
    """Install canonical structural state on ``g`` and rebuild every derived index.

    This installs a whole graph at once rather than changing one element. Copy,
    every subgraph, flatten and every loader arrive here, which is why it is the
    one place that fills the store of a graph outright.

    Pass ``store`` when the caller already holds the store this graph is to have,
    which copy and every selection do. Pass ``definitions`` instead when the
    caller parsed the graph from a file and holds no store of its own: it is the
    ``(entities, edges)`` pair :func:`store_from_definitions` takes.

    No caller hands over a matrix. The matrix of a graph is derived from the
    store it is given here, so a graph that never reads one never builds one,
    and one that does builds it from the store rather than from whatever the
    caller happened to hold.
    """
    from . import _mutate

    g._store = store_from_definitions(g, *definitions) if store is None else store
    g._mark_structure_changed()
    D.invalidate_sparse_caches(g)
    _mutate.sync_aspects(g)


def install_slices(g, slices, *, default=None, current=None) -> None:
    """Install the slice registry and (optionally) the default / active slice."""
    g._slices = slices
    if default is not None:
        g._default_slice = default
    if current is not None:
        g._current_slice = current
