"""The single graph-construction path.

Every code path that builds a fully-formed graph (copy, subgraph / flat-selection,
flatten-to-flat, IO load) installs its canonical state through here. Record construction
and structural-field assignment live in this module + ``_mutate``; consumers never assign
``_entities`` / ``_edges`` / ``_matrix`` or build records by hand.
"""

from __future__ import annotations

from . import _store as ST, _derive as D, _structure as S
from ._records import EdgeRecord, SliceRecord, EntityRecord

_SLOT_ENTITY_KIND = {S.NODE: ST.NODE, S.EDGE_ENTITY: ST.EDGE_ENTITY}
_SLOT_EDGE_KIND = {
    S.BINARY: ST.BINARY,
    S.HYPER: ST.HYPER,
    S.NODE_EDGE: ST.NODE_EDGE,
    S.PLACEHOLDER: ST.PLACEHOLDER,
}

# ---------------------------------------------------------------------------
# Record construction (the only place outside _mutate that builds records)
# ---------------------------------------------------------------------------


def new_entity_record(row_idx: int, kind: str) -> EntityRecord:
    """Build an :class:`EntityRecord` for a canonical entity row and kind."""
    return EntityRecord(row_idx=row_idx, kind=kind)


def clone_edge_record(rec, *, col_idx=None, weight=None) -> EdgeRecord:
    """Clone an EdgeRecord, optionally overriding col_idx / weight."""
    return EdgeRecord(
        src=rec.src,
        tgt=rec.tgt,
        weight=rec.weight if weight is None else weight,
        directed=rec.directed,
        etype=rec.etype,
        col_idx=rec.col_idx if col_idx is None else col_idx,
        ml_kind=rec.ml_kind,
        ml_layers=rec.ml_layers,
        direction_policy=rec.direction_policy,
        coeffs=dict(rec.coeffs) if rec.coeffs is not None else None,
    )


def clone_entities(src_entities) -> dict:
    """Clone an entity registry (preserving row indices)."""
    return {k: EntityRecord(row_idx=r.row_idx, kind=r.kind) for k, r in src_entities.items()}


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

    Nothing here reads a record, so this is what a load keeps once the records
    go. Returns None when the graph keeps records alone.
    """
    from . import _mutate

    if _mutate.slot_store(g) is None:
        return None

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
        )
    return store


def install_structure(
    g,
    *,
    entities,
    edges,
    matrix,
    store=None,
    definitions=None,
) -> None:
    """Install canonical structural state on ``g`` and rebuild every derived index.

    Pass ``matrix=None`` when the caller holds no materialized matrix. It then
    rebuilds from the store on the first read.

    Pass ``store`` when the caller already holds the slot store this graph is to
    have. Otherwise the store is rebuilt from what was just installed, which costs
    a pass over every edge, so a caller that can build the store itself should.

    Pass ``definitions`` instead when the caller parsed the graph from a file and
    holds no store of its own. It is the ``(entities, edges)`` pair
    :func:`store_from_definitions` takes, and the store is filled from it.
    """
    from . import _mutate

    if store is None and definitions is not None:
        store = store_from_definitions(g, *definitions)

    g._entities = entities
    g._edges = edges
    if matrix is None:
        g._mark_matrix_dirty()
    else:
        g._matrix = matrix
    D.rebuild_entity_indexes(g)  # _row_to_entity, _vid_to_ekeys
    D.rebuild_col_index(g)  # _col_to_edge
    D.invalidate_sparse_caches(g)
    # This installs a whole graph at once rather than changing one element, so the
    # slot store on it is installed or filled here. Copy, every subgraph and every
    # loader arrive here, which is why it is the one place that has to do it.
    if store is not None and _mutate.slot_store(g) is not None:
        g._store = store
    _mutate.sync_aspects(g)
    if store is None:
        _mutate.resync(g)


def install_slices(g, slices, *, default=None, current=None) -> None:
    """Install the slice registry and (optionally) the default / active slice."""
    g._slices = slices
    if default is not None:
        g._default_slice = default
    if current is not None:
        g._current_slice = current
