"""The single graph-construction path.

Every code path that builds a fully-formed graph (copy, subgraph / flat-selection,
flatten-to-flat, IO load) installs its canonical state through here. Record construction
and structural-field assignment live in this module + ``_mutate``; consumers never assign
``_entities`` / ``_edges`` / ``_matrix`` or build records by hand.
"""

from __future__ import annotations

from . import _derive as D
from ._records import EdgeRecord, SliceRecord, EntityRecord

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


def install_structure(g, *, entities, edges, matrix) -> None:
    """Install canonical structural state on ``g`` and rebuild every derived index."""
    g._entities = entities
    g._edges = edges
    g._matrix = matrix
    D.rebuild_entity_indexes(g)  # _row_to_entity, _vid_to_ekeys
    D.rebuild_col_index(g)  # _col_to_edge
    D.rebuild_edge_indexes(g)  # _src_to_edges, _tgt_to_edges, _pair_to_edges
    D.invalidate_sparse_caches(g)


def install_slices(g, slices, *, default=None, current=None) -> None:
    """Install the slice registry and (optionally) the default / active slice."""
    g._slices = slices
    if default is not None:
        g._default_slice = default
    if current is not None:
        g._current_slice = current
