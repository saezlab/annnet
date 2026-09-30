"""The internal attribute helpers mixed into ``AnnNet``.

The public attribute surface is :class:`annnet.core._attribute_api.Attrs`
(``G.attrs``): eight addresses, one shape. What remains here is what the
graph itself needs when an attribute changes — the flexible-direction
policies that re-orient an edge from a node or edge attribute — and the
audit of contextual rows against the structure.

Nothing here is a public route. The names are kept on the mixin so that the
mutation gateway and the attribute API can reach them without a second
object.
"""

from __future__ import annotations

from . import _structure
from ._state import GraphState

# The eight attribute tables, each named for the address it is keyed by.
TABLE_NAMES = (
    'nodes',
    'edges',
    'slices',
    'aspects',
    'layers',
    'edge_slices',
    'node_layers',
    'elementary_layers',
)


def _check_reserved_collision(reserved, attrs, *, kind, allow=()):
    if not attrs:
        return
    allow = set(allow)
    bad = sorted(k for k in attrs if k in reserved and k not in allow)
    if bad:
        raise ValueError(
            f'{kind} attributes use reserved key(s): {bad!r}. '
            f'These names are part of the structural / dispatch contract; '
            f'rename your attribute(s) to use a different key.'
        )


class AttributesClass(GraphState):
    """Flexible-direction policies and the attribute audit (mixed into AnnNet)."""

    def audit_attributes(self):
        """Audit the contextual rows against the structure. See ``G.attrs.audit``."""
        return self.attrs.audit()

    # ── flexible direction ─────────────────────────────────────────────────
    # An edge may declare that its direction follows an attribute: of itself
    # (``scope='edge'``) or of its two endpoints (``scope='node'``). A write to
    # that attribute re-resolves the edge. The attribute API calls these after
    # a batch has landed.

    def _variables_watched_by_nodes(self):
        return {
            p['var']
            for p in self.edge_direction_policy.values()
            if p.get('scope', 'edge') == 'node'
        }

    def _incident_flexible_edges(self, v):
        out = []
        policies = self.edge_direction_policy
        for ref in _structure.iter_edges(self):
            if ref.kind == _structure.HYPER or ref.id not in policies:
                continue
            sides = _structure.edge_sides(self, ref.id)
            if not sides.source or not sides.target:
                continue
            if v in sides.source or v in sides.target:
                out.append(ref.id)
        return out

    def _apply_flexible_direction(self, edge_id):
        pol = self.edge_direction_policy.get(edge_id)
        if not pol:
            return
        ref = _structure.edge_ref(self, edge_id)
        sides = _structure.edge_sides(self, edge_id)
        src = next(iter(sides.source), None)
        tgt = next(iter(sides.target), None)
        w = float(ref.weight if ref.weight is not None else 1.0)

        var = pol['var']
        T = float(pol['threshold'])
        scope = pol.get('scope', 'edge')
        above = pol.get('above', 's->t')
        tie = pol.get('tie', 'keep')

        tie_case = False
        if scope == 'edge':
            x = self._attr_store.edge_attr(edge_id, var, None)
            if x is None:
                return
            if x == T:
                tie_case = True
            cond = x > T
        else:
            xs = self._attr_store.node_attr(src, var, None)
            xt = self._attr_store.node_attr(tgt, var, None)
            if xs is None or xt is None:
                return
            if xs == xt:
                tie_case = True
            cond = (xs - xt) > 0

        # Persist the resolved column into the store so the lazily rebuilt
        # incidence matrix reflects it (the store is the source of truth).
        def _resolve(sval, tval):
            coeffs = {src: sval}
            if src != tgt:
                coeffs[tgt] = tval
            from . import _mutate

            _mutate.replace_edge_coeffs(self, edge_id, coeffs)
            self._mark_structure_changed()
            self._invalidate_sparse_caches()

        if tie_case:
            if tie == 'keep':
                return
            if tie == 'undirected':
                _resolve(+w, +w)
                return
            cond = True if tie == 's->t' else False

        src_to_tgt = cond if above == 's->t' else (not cond)
        if src_to_tgt:
            _resolve(+w, -w)
        else:
            _resolve(-w, +w)


__all__ = ['TABLE_NAMES', 'AttributesClass']
