"""The public attribute surface of a graph.

The generic attributes of a node and of an edge live in the column store of
``_attrs``, and everything here that reads or writes one goes through it. This
module holds no storage of its own for them.

The two contextual tables that are still frames — the attributes of a slice, and
the attributes of one edge inside one slice — are held by the graph and written
through the upsert helpers at the end of this module.
"""

import math
from typing import TYPE_CHECKING, Any

from . import _structure
from ._state import GraphState
from .._support.dataframe_backend import (
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_eq,
    dataframe_upsert_rows,
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
    """Attribute accessors and upsert helpers (graph/node/edge/slice/edge-slice)."""

    def set_graph_attribute(self, key, value):
        """Set a graph-level attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Attribute value.
        """
        self.graph_attributes[key] = value

    def get_graph_attribute(self, key, default=None):
        """Get a graph-level attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        default : Any, optional
            Value to return if the attribute is missing.

        Returns
        -------
        Any
        """
        return self.graph_attributes.get(key, default)

    def set_node_attrs(self, node_id, **attrs):
        """Upsert pure node attributes (non-structural) into the node table.

        Parameters
        ----------
        node_id : str
            Node identifier.
        **attrs
            Attribute key/value pairs.

        Raises
        ------
        ValueError
            If any key is structurally reserved (e.g. ``node_id``).
        """
        _check_reserved_collision(self._node_RESERVED, attrs, kind='node')
        clean = dict(attrs)
        if not clean:
            return

        if self._node_key_enabled():
            old_key = self._current_key_of_node(node_id)
            merged = {
                f: (
                    clean[f]
                    if f in clean
                    else AttributesClass.get_attr_node(self, node_id, f, None)
                )
                for f in self._node_key_fields
            }
            new_key = self._build_key_from_attrs(merged)
            if new_key is not None:
                owner = self._node_key_index.get(new_key)
                if owner is not None and owner != node_id:
                    raise ValueError(
                        f'Composite key collision on {self._node_key_fields}: {new_key} owned by {owner}'
                    )

        self._attr_store.set_node_attrs(node_id, clean)

        watched = self._variables_watched_by_nodes()
        if watched and any(k in watched for k in clean):
            for eid in self._incident_flexible_edges(node_id):
                self._apply_flexible_direction(eid)

        if self._node_key_enabled():
            new_key = self._current_key_of_node(node_id)
            old_key = old_key if 'old_key' in locals() else None
            if old_key != new_key:
                if old_key is not None and self._node_key_index.get(old_key) == node_id:
                    self._node_key_index.pop(old_key, None)
                if new_key is not None:
                    self._node_key_index[new_key] = node_id

    def set_node_attrs_bulk(self, updates):
        """Upsert node attributes in bulk.

        Parameters
        ----------
        updates : dict[str, dict] | Iterable[tuple[str, dict]]
            Mapping or iterable of `(node_id, attrs)` pairs.
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)
        for vid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(f'node bulk attrs must be dict, got {type(attrs)} for {vid}')
        for _vid, attrs in updates.items():
            _check_reserved_collision(self._node_RESERVED, attrs, kind='node')

        clean_updates = {vid: dict(attrs) for vid, attrs in updates.items() if attrs}
        if not clean_updates:
            return

        if self._node_key_enabled():
            old_keys = {vid: self._current_key_of_node(vid) for vid in clean_updates}
            new_keys = {}
            for vid, attrs in clean_updates.items():
                merged = {
                    f: (
                        attrs[f]
                        if f in attrs
                        else AttributesClass.get_attr_node(self, vid, f, None)
                    )
                    for f in self._node_key_fields
                }
                new_keys[vid] = self._build_key_from_attrs(merged)
            for vid, new_key in new_keys.items():
                if new_key is not None:
                    owner = self._node_key_index.get(new_key)
                    if owner is not None and owner != vid:
                        raise ValueError(
                            f'Composite key collision on {self._node_key_fields}: {new_key} owned by {owner}'
                        )

        for vid, attrs in clean_updates.items():
            self._attr_store.set_node_attrs(vid, attrs)

        watched = self._variables_watched_by_nodes()
        if watched:
            affected_nodes = {
                vid for vid, attrs in clean_updates.items() if any(k in watched for k in attrs)
            }
            if affected_nodes:
                affected_edges = set()
                for vid in affected_nodes:
                    affected_edges.update(self._incident_flexible_edges(vid))
                for eid in affected_edges:
                    self._apply_flexible_direction(eid)

        if self._node_key_enabled():
            for vid in clean_updates:
                new_key, old_key = new_keys[vid], old_keys[vid]
                if old_key != new_key:
                    if old_key is not None and self._node_key_index.get(old_key) == vid:
                        self._node_key_index.pop(old_key, None)
                    if new_key is not None:
                        self._node_key_index[new_key] = vid

    def get_attr_node(self, node_id, key, default=None):
        """Get a single node attribute (scalar) or default if missing.

        Parameters
        ----------
        node_id : str
            Node identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._attr_store.node_attr(node_id, key, default)

    def set_edge_attrs(self, edge_id, **attrs):
        """Upsert pure edge attributes (non-structural) into the edge DF.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        **attrs
            Attribute key/value pairs.

        Raises
        ------
        ValueError
            If any key is structurally reserved (e.g. ``edge_id``,
            ``source``, ``target``, ``weight``, ``members``, ``head``,
            ``tail``, ``flexible``).
        """
        _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge')
        clean = dict(attrs)
        if clean:
            self._attr_store.set_edge_attrs(edge_id, clean)
        pol = self.edge_direction_policy.get(edge_id)
        if pol and pol.get('scope', 'edge') == 'edge' and pol['var'] in clean:
            self._apply_flexible_direction(edge_id)

    def set_edge_attrs_bulk(self, updates):
        """Upsert edge attributes in bulk.

        Parameters
        ----------
        updates : dict[str, dict] | Iterable[tuple[str, dict]]
            Mapping or iterable of `(edge_id, attrs)` pairs.
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)
        for eid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(f'edge bulk attrs must be dict, got {type(attrs)} for {eid}')
        for _eid, attrs in updates.items():
            _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge')

        clean_updates = {eid: dict(attrs) for eid, attrs in updates.items() if attrs}
        if not clean_updates:
            return

        for eid, attrs in clean_updates.items():
            self._attr_store.set_edge_attrs(eid, attrs)

        policy_map = self.edge_direction_policy
        affected_edges = set()
        for eid, attrs in clean_updates.items():
            pol = policy_map.get(eid)
            if pol and pol.get('scope') == 'edge' and pol['var'] in attrs:
                affected_edges.add(eid)
        for eid in affected_edges:
            self._apply_flexible_direction(eid)

    def get_attr_edge(self, edge_id, key, default=None):
        """Get a single edge attribute (scalar) or default if missing.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._attr_store.edge_attr(edge_id, key, default)

    def set_slice_attrs(self, slice_id, **attrs):
        """Upsert pure slice attributes.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored.
        """
        _check_reserved_collision(self._slice_RESERVED, attrs, kind='slice')
        clean = dict(attrs)
        if clean:
            self.slice_attributes = self._upsert_row(self.slice_attributes, slice_id, clean)

    def get_slice_attr(self, slice_id, key, default=None):
        """Get a single slice attribute (scalar) or default if missing.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        df = self.slice_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'slice_id', slice_id))
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

    def set_edge_slice_attrs(self, slice_id, edge_id, **attrs):
        """Upsert per-slice attributes for a specific edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored except `weight`.
        """
        _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge-slice', allow=('weight',))
        clean = dict(attrs)
        if not clean:
            return
        try:
            import sys as _sys

            if isinstance(slice_id, str):
                slice_id = _sys.intern(slice_id)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except (AttributeError, TypeError):
            pass
        if 'weight' in clean:
            try:
                clean['weight'] = float(clean['weight'])
            except (TypeError, ValueError):
                pass
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )
        self._sync_slice_edge_weights_for_rows(
            slice_id, [{'slice_id': slice_id, 'edge_id': edge_id, **clean}]
        )

    def edge_slice(self, slice_id, edge_id) -> dict:
        """Return every attribute one edge carries in one slice.

        The level of this store is the pair, so an edge that carries nothing in
        this slice answers with an empty mapping rather than with the
        attributes it carries elsewhere.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.

        Returns
        -------
        dict
        """
        df = self.edge_slice_attributes
        if df is None:
            return {}
        for row in dataframe_to_rows(df):
            if row.get('slice_id') == slice_id and row.get('edge_id') == edge_id:
                return {
                    key: value
                    for key, value in row.items()
                    if key not in ('slice_id', 'edge_id') and value is not None
                }
        return {}

    def get_edge_slice_attr(self, slice_id, edge_id, key, default=None):
        """Get a per-slice attribute for an edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        df = self.edge_slice_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = [
            row
            for row in dataframe_to_rows(df)
            if row.get('slice_id') == slice_id and row.get('edge_id') == edge_id
        ]
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

    def set_slice_edge_weight(self, slice_id, edge_id, weight):
        """Set a legacy per-slice weight override for an edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        weight : float
            Weight override.

        Raises
        ------
        KeyError
            If the slice or edge does not exist.

        See Also
        --------
        get_effective_edge_weight
        """
        if slice_id not in self._slices:
            raise KeyError(f'slice {slice_id} not found')
        if not _structure.has_edge(self, edge_id) or not _structure.carries_structure(
            self, edge_id
        ):
            raise KeyError(f'Edge {edge_id} not found')
        AttributesClass.set_edge_slice_attrs(self, slice_id, edge_id, weight=float(weight))

    def get_effective_edge_weight(self, edge_id, slice=None):
        """Resolve the effective weight for an edge, optionally within a slice.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        slice : str, optional
            Slice to read the override from. When omitted, the graph's
            currently active slice is used. Pass an explicit slice ID to
            override the active-slice resolution.

        Returns
        -------
        float
            Effective weight.
        """
        if slice is None:
            slice = self._current_slice
        if slice is not None:
            df = self.edge_slice_attributes
            if df is not None and {'slice_id', 'edge_id', 'weight'} <= set(dataframe_columns(df)):
                for row in dataframe_to_rows(df):
                    if row.get('slice_id') != slice or row.get('edge_id') != edge_id:
                        continue
                    w = row.get('weight', None)
                    if w is not None and not (isinstance(w, float) and math.isnan(w)):
                        return float(w)
        if not _structure.has_edge(self, edge_id):
            return 1.0
        return float(_structure.edge_ref(self, edge_id).weight)

    def audit_attributes(self):
        """Audit attribute tables for extra/missing rows and invalid edge-slice pairs.

        Returns
        -------
        dict
            Summary with keys:
            - `extra_node_rows`
            - `extra_edge_rows`
            - `missing_node_rows`
            - `missing_edge_rows`
            - `invalid_edge_slice_rows`

        Notes
        -----
        The node table and the edge table are derived from columns addressed by
        slot, so a row of either names an element the graph holds and every
        element the graph holds has one. The first four lists are therefore
        always empty, and what this still finds is an edge-by-slice row that
        names a slice or an edge the graph does not hold.
        """
        node_ids = {ref.id for ref in _structure.iter_entities(self) if ref.kind == _structure.NODE}
        edge_ids = {ref.id for ref in _structure.iter_edges(self)}
        ela = self.edge_slice_attributes

        node_attr_ids = set(self._attr_store.node_ids())
        edge_attr_ids = set(self._attr_store.edge_ids())

        bad_edge_slice = []
        if ela is not None and {'slice_id', 'edge_id'} <= set(dataframe_columns(ela)):
            for r in dataframe_to_rows(ela):
                lid, eid = r.get('slice_id'), r.get('edge_id')
                if lid not in self._slices or eid not in edge_ids:
                    bad_edge_slice.append((lid, eid))

        return {
            'extra_node_rows': [i for i in node_attr_ids if i not in node_ids],
            'extra_edge_rows': [i for i in edge_attr_ids if i not in edge_ids],
            'missing_node_rows': [i for i in node_ids if i not in node_attr_ids],
            'missing_edge_rows': [i for i in edge_ids if i not in edge_attr_ids],
            'invalid_edge_slice_rows': bad_edge_slice,
        }

    # ── the two contextual tables ────────────────────────────────────────────
    # The slice table and the edge-by-slice table are frames still, keyed by the
    # level they belong to. Their rows are written whole, so a write states the
    # key columns and the values together.

    def _upsert_row(self, df: 'object', idx: Any, attrs: dict) -> 'object':
        if not isinstance(attrs, dict) or not attrs:
            return df
        cols = set(dataframe_columns(df))
        key_cols: tuple[str, ...]
        if {'slice_id', 'edge_id'} <= cols:
            key_cols = ('slice_id', 'edge_id')
            key_vals = {'slice_id': idx[0], 'edge_id': idx[1]}
        elif 'slice_id' in cols:
            key_cols = ('slice_id',)
            key_vals = {'slice_id': idx}
        else:
            raise ValueError('Cannot infer key columns from DataFrame schema')
        return dataframe_upsert_rows(df, [{**key_vals, **attrs}], key_cols)

    def _upsert_rows_bulk(self, df: 'object', updates: dict) -> 'object':
        if not updates:
            return df
        cols = set(dataframe_columns(df))
        join_keys = ('slice_id', 'edge_id') if {'slice_id', 'edge_id'} <= cols else ('slice_id',)
        update_records = [
            {'slice_id': idx[0], 'edge_id': idx[1], **attrs}
            if isinstance(idx, tuple)
            else {'slice_id': idx, **attrs}
            for idx, attrs in updates.items()
        ]
        return dataframe_upsert_rows(df, update_records, join_keys)

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
            x = AttributesClass.get_attr_edge(self, edge_id, var, None)
            if x is None:
                return
            if x == T:
                tie_case = True
            cond = x > T
        else:
            xs = AttributesClass.get_attr_node(self, src, var, None)
            xt = AttributesClass.get_attr_node(self, tgt, var, None)
            if xs is None or xt is None:
                return
            if xs == xt:
                tie_case = True
            cond = (xs - xt) > 0

        # Persist the resolved column into the record's coeffs so the lazily
        # rebuilt incidence matrix reflects it (records are the source of truth).
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

    # ── full / bulk reads ─────────────────────────────────────────────────────

    def get_edge_attrs(self, edge) -> dict:
        """Return the full attribute dict for a single edge.

        Parameters
        ----------
        edge : int | str
            Edge index or edge ID.

        Returns
        -------
        dict
            Attribute dictionary for that edge. Empty if not found.
        """
        eid = _structure.edge_at_column(self, edge) if isinstance(edge, int) else edge
        return self._attr_store.edge_attrs(eid)

    def get_node_attrs(self, node) -> dict:
        """Return the full attribute dict for a single node.

        Parameters
        ----------
        node : str
            Node ID.

        Returns
        -------
        dict
            Attribute dictionary for that node. Empty if not found.
        """
        return self._attr_store.node_attrs(node)

    def get_attr_edges(self, indexes=None) -> dict:
        """Retrieve edge attributes as a dictionary.

        Parameters
        ----------
        indexes : Iterable[int] | None, optional
            Edge indices to retrieve. If None, returns all edges.

        Returns
        -------
        dict[str, dict]
            Mapping of `edge_id` to attribute dictionaries.
        """
        rows = dataframe_to_rows(self._edge_table)
        if indexes is not None:
            wanted = {_structure.edge_at_column(self, i) for i in indexes}
            rows = [row for row in rows if row.get('edge_id') in wanted]
        return {r.get('edge_id'): dict(r) for r in rows if r.get('edge_id') is not None}

    def get_attr_nodes(self, nodes=None) -> dict:
        """Retrieve node (node) attributes as a dictionary.

        Parameters
        ----------
        nodes : Iterable[str] | None, optional
            Node IDs to retrieve. If None, returns all nodes.

        Returns
        -------
        dict[str, dict]
            Mapping of `node_id` to attribute dictionaries.
        """
        rows = dataframe_to_rows(self._node_table)
        if nodes is not None:
            wanted = set(nodes)
            rows = [row for row in rows if row.get('node_id') in wanted]
        return {r.get('node_id'): dict(r) for r in rows if r.get('node_id') is not None}

    def get_attr_from_edges(self, key: str, default=None) -> dict:
        """Extract a specific attribute column for all edges.

        Parameters
        ----------
        key : str
            Attribute column name to extract.
        default : Any, optional
            Value to use if the column or value is missing.

        Returns
        -------
        dict[str, Any]
            Mapping of `edge_id` to attribute values.
        """
        column = self._attr_store.edge_attr_map(key)
        if column is None:
            return dict.fromkeys(self._attr_store.edge_ids(), default)
        return {edge_id: (default if value is None else value) for edge_id, value in column.items()}

    def get_edges_by_attr(self, key: str, value) -> list:
        """Retrieve all edges where a given attribute equals a specific value.

        Parameters
        ----------
        key : str
            Attribute column name to filter on.
        value : Any
            Value to match.

        Returns
        -------
        list[str]
            Edge IDs where the attribute equals `value`.
        """
        column = self._attr_store.edge_attr_map(key)
        if column is None:
            return []
        return [edge_id for edge_id, held in column.items() if held == value]

    def get_graph_attributes(self) -> dict:
        """Return a shallow copy of the graph-level attributes dictionary.

        Returns
        -------
        dict
            Shallow copy of global graph metadata.

        Notes
        -----
        Returned value is a shallow copy to prevent external mutation.
        """
        return dict(self.graph_attributes)

    def set_edge_slice_attrs_bulk(self, slice_id, items):
        """Upsert edge-slice attributes for a single slice in bulk.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        items : Iterable[tuple[str, dict]] | dict[str, dict]
            Iterable or mapping of `(edge_id, attrs)` pairs.
        """
        rows = []
        it = items.items() if isinstance(items, dict) else items
        for eid, attrs in it:
            if not isinstance(attrs, dict) or not attrs:
                continue
            r = {'slice_id': slice_id, 'edge_id': eid, **attrs}
            if 'weight' in r:
                try:
                    r['weight'] = float(r['weight'])
                except (TypeError, ValueError):
                    pass
            rows.append(r)
        if not rows:
            return
        updates = {
            (row['slice_id'], row['edge_id']): {
                k: v for k, v in row.items() if k not in {'slice_id', 'edge_id'}
            }
            for row in rows
        }
        self.edge_slice_attributes = self._upsert_rows_bulk(self.edge_slice_attributes, updates)
        self._sync_slice_edge_weights_for_rows(slice_id, rows)


# Methods exposed verbatim on the ``G.attrs`` namespace.
_ATTR_DELEGATED = (
    'set_graph_attribute',
    'get_graph_attribute',
    'get_graph_attributes',
    'set_node_attrs',
    'set_node_attrs_bulk',
    'get_node_attrs',
    'get_attr_node',
    'get_attr_nodes',
    'set_edge_attrs',
    'set_edge_attrs_bulk',
    'get_edge_attrs',
    'get_attr_edge',
    'get_attr_edges',
    'get_attr_from_edges',
    'get_edges_by_attr',
    'set_slice_attrs',
    'get_slice_attr',
    'set_edge_slice_attrs',
    'set_edge_slice_attrs_bulk',
    'get_edge_slice_attr',
    'edge_slice',
    'set_slice_edge_weight',
    'get_effective_edge_weight',
    'audit_attributes',
)


class AttributesAccessor:
    """Namespace for graph, node, edge, and slice annotations (``G.attrs``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    if TYPE_CHECKING:  # pragma: no cover - the delegators are installed below

        def __getattr__(self, name: str) -> Any:
            """Every name of ``_ATTR_DELEGATED``, installed at import."""


def _install_attr_delegators():
    for _name in _ATTR_DELEGATED:

        def _make(name):
            target = getattr(AttributesClass, name)

            def _delegator(self, *args, **kwargs):
                return target(self._G, *args, **kwargs)

            _delegator.__name__ = name
            _delegator.__qualname__ = f'AttributesAccessor.{name}'
            _delegator.__doc__ = target.__doc__
            return _delegator

        setattr(AttributesAccessor, _name, _make(_name))


_install_attr_delegators()
