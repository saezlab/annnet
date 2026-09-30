"""The eight attribute addresses, read and written one way (``G.attrs``).

An *address* is what a set of attributes is keyed by: a node, an edge, a slice,
an aspect, a layer coordinate, an edge inside a slice, a node inside a layer, or
a label inside an aspect. Eight of them, and every one answers the same shape.

Reading always gives a detached copy; editing what comes back changes nothing
the graph holds:

- ``G.attrs.nodes`` — the stored table, as a dataframe. ``G.attrs.table(address,
  ...)`` is the same read for an address held in a variable, with the backend,
  columns, a preview limit and the derived fields as arguments;
- ``G.attrs.row('nodes', 'A')`` — one row, a dict; ``G.attrs.rows('nodes',
  keys)`` — many, ``{key: dict}``. A cell is ``row(...).get(name, default)``;
- ``G.attrs.schema('edges')`` — the key and value fields without a frame.

Writing names what it does and is checked as a whole before it changes anything:

- ``G.attrs.update('nodes', {...})`` merges the fields it names into rows;
- ``G.attrs.replace('nodes', table)`` replaces the whole address with a table;
- ``G.attrs.delete('nodes', keys=..., names=...)`` removes attributes and never
  the nodes, edges or other structure that carry them.

Selecting comes in two kinds that do not mix. ``G.attrs.select('nodes',
score__gt=0.5)`` is a live query over the graph, as ``G.N.select`` is: it keeps
its conditions and answers again after the graph changes. Filtering a table you
have read is the dataframe library's job; ``G.attrs.from_frame('nodes', frame)``
turns the rows that are left into a fixed selection of the graph's keys.

This module owns the public *addressing* of attributes: which keys an address
accepts, how they are normalized and validated, what each address's domain is,
and how a row or a table is read and written. It delegates:

- predicate parsing to :mod:`annnet.core._predicate` and row selections to
  :class:`annnet.core._select.RowSelection`, which calls back into the
  reader protocol here (``_domain``, ``_rows_where``, ``_row``, ...);
- the all-or-nothing guarantee to
  :class:`annnet.core._transaction.AttributeTransaction`;
- derived frames to :mod:`annnet.core._tables`.

The generic node and edge attributes live in the slot-indexed column store of
:mod:`annnet.core._attrs`. The six contextual addresses live in the one
:class:`~annnet.core._contextual.ContextualStore`. This module owns neither.

**A null value in an update removes the field.** Writing ``None`` (or a float
NaN) for a field in ``update`` removes it from the row, and a table cell that is
null means the same. ``delete`` is the explicit way to say the same thing. This
is one rule for every address and every backend.

A view scopes the same object: :class:`ScopedAttrs` reads the same eight
addresses restricted to the elements the view resolved to, and refuses every
write with a message naming ``.materialize()``.
"""

from __future__ import annotations

import copy
from typing import NamedTuple
import itertools
from collections.abc import Mapping

import numpy as np

from . import _structure
from ._attrs import EDGE_AXIS, NODE_AXIS
from ._select import Leaf, RowSelection, ReadOnlyViewError, project_keys, refuse_write
from ._aspects import ORDERED_KEY
from ._records import _EDGE_RESERVED, _node_RESERVED, _slice_RESERVED
from ._predicate import rows_matching, parse_conditions
from ._transaction import AttributeTransaction
from .._support.dataframe_backend import (
    clone_dataframe,
    empty_dataframe,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_from_rows,
    select_dataframe_backend,
)

#: The eight addresses, in the order they are listed everywhere.
ADDRESSES = (
    'nodes',
    'edges',
    'slices',
    'aspects',
    'layers',
    'edge_slices',
    'node_layers',
    'elementary_layers',
)

#: The key columns of each address, in the order the table shows them.
KEY_COLUMNS = {
    'nodes': ('node_id',),
    'edges': ('edge_id',),
    'slices': ('slice_id',),
    'aspects': ('aspect',),
    'layers': ('layer',),
    'edge_slices': ('slice_id', 'edge_id'),
    'node_layers': ('node_id', 'layer'),
    'elementary_layers': ('aspect', 'elementary_layer'),
}

#: The contextual store level behind each contextual address.
LEVEL_OF = {
    'slices': 'slice_attrs',
    'aspects': 'aspect_attrs',
    'layers': 'layer_attrs',
    'edge_slices': 'edge_slice_attrs',
    'node_layers': 'node_layer_attrs',
    'elementary_layers': 'elementary_attrs',
}

_GENERIC = ('nodes', 'edges')

#: The names an address reserves: its own key columns, plus the structural
#: fields of an edge. ``weight`` is allowed on ``edge_slices``, because a
#: per-slice weight override is exactly what that address is for.
_RESERVED = {
    'nodes': frozenset(_node_RESERVED),
    'edges': frozenset(_EDGE_RESERVED),
    'slices': frozenset(_slice_RESERVED),
    'aspects': frozenset({'aspect', ORDERED_KEY}),
    'layers': frozenset({'layer', 'layer_id', 'coordinate_id'}),
    'edge_slices': frozenset((_EDGE_RESERVED - {'weight'}) | {'slice_id'}),
    'node_layers': frozenset({'node_id', 'layer', 'layer_id'}),
    'elementary_layers': frozenset({'aspect', 'elementary_layer', 'layer_id'}),
}

#: Fields a level stores beside the attributes that are declarations of the
#: registry, not attributes a caller wrote: read through their own API
#: (``G.layers.aspect(name).ordered``), never shown as a row field, never
#: written through this one, and kept through a row or table replacement.
_DECLARATIONS = {'aspects': frozenset({ORDERED_KEY})}


def _declared(address) -> frozenset:
    return _DECLARATIONS.get(address, frozenset())


_SCHEMA_TYPE_OF_KEY = {
    'node_id': 'str',
    'edge_id': 'str',
    'slice_id': 'str',
    'aspect': 'str',
    'layer': 'tuple[str]',
    'elementary_layer': 'str',
}


_ATOMS = (str, bytes, int, float, complex, bool, type(None), np.generic)


def _detached(value):
    """A value the caller may edit without reaching the graph.

    Scalars are immutable and are returned as they are. A container (a list, a
    dict, a set, an array) is copied whole, because a row that handed out the
    list the store holds would let ``row['tags'].append(...)`` change the graph
    without a validation, a clock or a history entry.
    """
    return value if isinstance(value, _ATOMS) else copy.deepcopy(value)


def _holds_containers(column) -> bool:
    return column.dtype.kind == 'O' and any(
        not isinstance(v, _ATOMS) for v in column if v is not None
    )


def _is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    return False


def _coordinate(value) -> tuple:
    """Return one layer coordinate as a tuple of labels."""
    if isinstance(value, str):
        return (value,)
    return tuple(str(part) if not isinstance(part, str) else part for part in value)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class Schema(NamedTuple):
    """What one address is keyed by and what it holds, without building a frame."""

    address: str
    keys: tuple
    fields: tuple
    rows: int
    derived: bool = False

    def __repr__(self) -> str:
        keys = ', '.join(f'{name}: {kind}' for name, kind in self.keys)
        fields = ', '.join(f'{name}: {kind}' for name, kind in self.fields)
        head = f'Schema({self.address!r}{", derived" if self.derived else ""}, {self.rows} row(s))'
        return f'{head}\n  keys:   {keys or "-"}\n  fields: {fields or "-"}'

    def names(self) -> list[str]:
        """Every column name, keys first."""
        return [name for name, _kind in self.keys] + [name for name, _kind in self.fields]


def _type_name(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return 'bool'
    if isinstance(value, (int, np.integer)):
        return 'int'
    if isinstance(value, (float, np.floating)):
        return 'float'
    if isinstance(value, str):
        return 'str'
    if isinstance(value, (list, tuple, np.ndarray)):
        return 'list'
    if isinstance(value, dict):
        return 'dict'
    return type(value).__name__


def _column_type(column: np.ndarray) -> str:
    kind = column.dtype.kind
    if kind == 'f':
        return 'float'
    if kind in 'iu':
        return 'int'
    if kind == 'b':
        return 'bool'
    seen = None
    for value in column:
        if _is_null(value):
            continue
        name = _type_name(value)
        if seen is None:
            seen = name
        elif seen != name:
            return 'object'
    return seen or 'object'


# ---------------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------------


class Attrs:
    """Every attribute of a graph, at eight addresses (``G.attrs``).

    See the module docstring for the shape. The reader protocol the selection
    machinery uses is the set of underscored methods ``_domain``, ``_row``,
    ``_rows_where``, ``_key``, ``_selection_owner`` and ``_root_reader``.
    """

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    # -- scope (a graph is unscoped) --------------------------------------------

    @property
    def _scope(self):
        return None

    def _writable(self, what: str = 'a view') -> None:
        return None

    def _selection_owner(self):
        """The object a projected selection is evaluated in."""
        return self._G

    def _root_reader(self):
        """The unscoped reader of the same graph."""
        return self._G.attrs

    # -- the namespace protocol ---------------------------------------------

    def __iter__(self):
        return iter(ADDRESSES)

    def __len__(self):
        return len(ADDRESSES)

    def __contains__(self, address):
        return address in ADDRESSES

    def __dir__(self):
        return sorted(
            (
                *ADDRESSES,
                'audit',
                'backend',
                'delete',
                'from_frame',
                'replace',
                'row',
                'rows',
                'schema',
                'select',
                'table',
                'update',
            )
        )

    def __repr__(self):
        counts = ', '.join(f'{address}={len(self._domain(address))}' for address in ADDRESSES)
        scope = ' (view)' if self._scope is not None else ''
        return f'Attrs{scope}[{counts}]'

    # -- the eight tables -----------------------------------------------------

    @property
    def backend(self):
        """The dataframe backend every table is rendered in."""
        return self._G._annotations_backend

    @backend.setter
    def backend(self, value):
        self._writable('the backend of a view')
        self._G._annotations_backend = select_dataframe_backend(value)

    # -- what an attribute address is not ------------------------------------------
    #
    # An address is not indexed. Indexing used to hand back a live row that wrote
    # through to the graph and let a table be assigned, which made an edit of a
    # returned value change the graph. Each of those is now a named call.

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            address, key = item
            raise TypeError(
                f'G.attrs[{address!r}, {key!r}] is not supported; read the row with '
                f'G.attrs.row({address!r}, {key!r}), which is a detached dict'
            )
        if isinstance(item, str):
            raise TypeError(
                f'G.attrs[{item!r}] is not supported; read the table with '
                f'G.attrs.{item} or G.attrs.table({item!r})'
            )
        raise TypeError('G.attrs is not indexed; use row(), rows() or table()')

    def __setitem__(self, item, value):
        self._writable()
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            address, key = item
            raise TypeError(
                f'G.attrs[{address!r}, {key!r}] = ... is not supported; merge fields with '
                f'G.attrs.update({address!r}, {{{key!r}: {{...}}}})'
            )
        if isinstance(item, str):
            raise TypeError(
                f'G.attrs[{item!r}] = ... is not supported; replace the table with '
                f'G.attrs.replace({item!r}, table)'
            )
        raise TypeError('G.attrs is not indexed; write with update(), replace() or delete()')

    def __delitem__(self, item):
        self._writable()
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            address, key = item
            raise TypeError(
                f'del G.attrs[{address!r}, {key!r}] is not supported; clear the row with '
                f'G.attrs.delete({address!r}, keys=[{key!r}])'
            )
        raise TypeError('G.attrs is not indexed; remove attributes with delete()')

    # -- addresses and keys --------------------------------------------------

    def _address(self, address) -> str:
        if address not in ADDRESSES:
            raise KeyError(
                f'unknown attribute address {address!r}; expected one of {list(ADDRESSES)}'
            )
        return address

    def _elementary_from_id(self, layer_id: str) -> tuple:
        """Decode a legacy ``aspect_label`` id, refusing an ambiguous one."""
        g = self._G
        found = []
        for aspect in g._aspects:
            if aspect == '_':
                continue
            prefix = f'{aspect}_'
            if layer_id.startswith(prefix):
                label = layer_id[len(prefix) :]
                if label in g._layers.get(aspect, ()):
                    found.append((aspect, label))
        if len(found) == 1:
            return found[0]
        if not found:
            raise KeyError(f'unknown elementary layer id {layer_id!r}')
        raise ValueError(
            f'ambiguous elementary layer id {layer_id!r}: it could be any of {found!r}; '
            f'use an explicit (aspect, label) key'
        )

    def _in_scope(self, axis: str, key) -> bool:
        scope = self._scope
        if scope is None:
            return True
        return scope.holds(axis, key)

    def _require_scope(self, axis: str, key, what: str) -> None:
        if not self._in_scope(axis, key):
            raise KeyError(f'{what} is outside this view')

    def _key(self, address, key):
        """Normalize and validate one row key of one address."""
        self._address(address)
        g = self._G
        if address == 'nodes':
            if not isinstance(key, str):
                raise TypeError(f'a node key is a node id, not {type(key).__name__}')
            if not g.has_node(key):
                raise KeyError(f'unknown node {key!r}')
            self._require_scope('nodes', key, f'node {key!r}')
            return key
        if address == 'edges':
            if not isinstance(key, str):
                raise TypeError(f'an edge key is an edge id, not {type(key).__name__}')
            if not _structure.has_edge(g, key):
                raise KeyError(f'unknown edge {key!r}')
            self._require_scope('edges', key, f'edge {key!r}')
            return key
        if address == 'slices':
            if not isinstance(key, str):
                raise TypeError(f'a slice key is a slice id, not {type(key).__name__}')
            if key not in g._slices:
                raise KeyError(f'unknown slice {key!r}')
            self._require_scope('slices', key, f'slice {key!r}')
            return key
        if address == 'aspects':
            if not isinstance(key, str):
                raise TypeError(f'an aspect key is an aspect name, not {type(key).__name__}')
            if g._aspects == ('_',) or key not in g._aspects:
                raise KeyError(f'unknown aspect {key!r}; this graph declares {list(g.aspects)!r}')
            return key
        if address == 'layers':
            if g._aspects == ('_',):
                raise ValueError(
                    'no aspects are configured; declare them with G.layers.set_aspects'
                )
            coordinate = _coordinate(key)
            g.layers._validate_layer_tuple(coordinate)
            self._require_scope('layers', coordinate, f'layer {coordinate!r}')
            return coordinate
        if address == 'node_layers':
            if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str)):
                raise TypeError(
                    'a node-layer key is (node_id, layer); a bare node id needs layer= in update()'
                )
            node_id, coordinate = key
            coordinate = ('_',) if g._aspects == ('_',) else _coordinate(coordinate)
            placement = (node_id, coordinate)
            if not g._has_node_layer(placement):
                raise KeyError(f'{node_id!r} is not placed on layer {coordinate!r}')
            self._require_scope('node_layers', placement, f'placement {placement!r}')
            return placement
        if address == 'edge_slices':
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError('an edge-slice key is (slice_id, edge_id)')
            slice_id, edge_id = key
            if slice_id not in g._slices:
                raise KeyError(f'unknown slice {slice_id!r}')
            if not _structure.has_edge(g, edge_id):
                raise KeyError(f'unknown edge {edge_id!r}')
            self._require_scope('slices', slice_id, f'slice {slice_id!r}')
            self._require_scope('edges', edge_id, f'edge {edge_id!r}')
            return (slice_id, edge_id)
        # elementary_layers
        if g._aspects == ('_',):
            raise ValueError('no aspects are configured; declare them with G.layers.set_aspects')
        if isinstance(key, str):
            key = self._elementary_from_id(key)
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError('an elementary layer key is (aspect, label)')
        aspect, label = key
        if aspect not in g._aspects:
            raise KeyError(f'unknown aspect {aspect!r}; this graph declares {list(g.aspects)!r}')
        if label not in g._layers.get(aspect, ()) or label == '_':
            raise KeyError(f'unknown elementary layer {label!r} for aspect {aspect!r}')
        self._require_scope('elementary_layers', (aspect, label), f'elementary layer {key!r}')
        return (aspect, label)

    # -- the domain of each address --------------------------------------------

    def _domain(self, address) -> list:
        """The keys an address addresses, in the address's own order.

        Nodes, edges, slices, aspects and elementary layers are the structural
        registries. Layers are the coordinates that occur — a placement or a
        stored row — in declaration order. Node-layers are the placements in
        row order. Edge-slices are the explicit memberships plus the stored
        override pairs, slice by slice and edge by edge; never a product.
        """
        g = self._G
        scope = self._scope
        if address == 'nodes':
            return list(scope.node_ids) if scope is not None else _structure.node_ids(g)
        if address == 'edges':
            return list(scope.edge_ids) if scope is not None else _structure.edge_ids(g)
        if address == 'slices':
            ids = list(g._slices)
            return [s for s in ids if scope.holds('slices', s)] if scope is not None else ids
        if address == 'aspects':
            return [] if g._aspects == ('_',) else list(g._aspects)
        if address == 'layers':
            if g._aspects == ('_',):
                return []
            occurring = {key[1] for key in _structure.node_keys(g)}
            occurring.update(g._contextual.layer_attrs)
            declared = [tuple(aa) for aa in g.layers._all_layers]
            found = [aa for aa in declared if aa in occurring]
            if scope is not None:
                found = [aa for aa in found if scope.holds('layers', aa)]
            return found
        if address == 'node_layers':
            return list(scope.node_keys) if scope is not None else _structure.node_keys(g)
        if address == 'edge_slices':
            stored = g._contextual.edge_slice_attrs
            edge_order = {eid: n for n, eid in enumerate(self._domain('edges'))}
            out: list = []
            for slice_id in self._domain('slices'):
                members = set(g._slices[slice_id].edges)
                present = {eid for eid in members if eid in edge_order}
                present.update(
                    eid for (sid, eid) in stored if sid == slice_id and eid in edge_order
                )
                out.extend((slice_id, eid) for eid in sorted(present, key=edge_order.__getitem__))
            return out
        # elementary_layers
        if g._aspects == ('_',):
            return []
        out = []
        for aspect in g._aspects:
            for label in g._layers.get(aspect, ()):
                if label != '_':
                    out.append((aspect, label))
        if scope is not None:
            out = [key for key in out if scope.holds('elementary_layers', key)]
        return out

    # -- reading rows ------------------------------------------------------------

    def _row(self, address, key) -> dict:
        """The stored attributes of one validated key, as a fresh dict."""
        g = self._G
        if address == 'nodes':
            return {k: _detached(v) for k, v in g._attr_store.node_attrs(key).items()}
        if address == 'edges':
            return {k: _detached(v) for k, v in g._attr_store.edge_attrs(key).items()}
        held = getattr(g._contextual, LEVEL_OF[address]).get(key)
        if not held:
            return {}
        hidden = _declared(address)
        return {
            name: _detached(value)
            for name, value in held.items()
            if name not in hidden and not _is_null(value)
        }

    def row(self, address, key) -> dict:
        """Return one row as a detached dict.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        key : key
            The key of the row, as the address keys it: a node id, an edge id, a
            slice id, an aspect name, a layer coordinate, a ``(slice_id,
            edge_id)`` pair, a ``(node_id, layer)`` placement or an ``(aspect,
            label)`` pair.

        Returns
        -------
        dict
            ``{name: value}``; ``{}`` when the element carries no attribute.
            Editing it changes nothing in the graph. A cell is
            ``row(address, key).get(name, default)``.

        Raises
        ------
        KeyError
            For a key the address does not hold (or, on a view, one outside it).
        """
        self._address(address)
        return self._row(address, self._key(address, key))

    def rows(self, address, keys=None) -> dict:
        """Return detached row dictionaries keyed by the address's keys.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        keys : key | Iterable[key], optional
            One key or many. Default: every key of the address.

        Returns
        -------
        dict
            ``{key: {name: value}}``. A key that carries nothing maps to ``{}``.
        """
        self._address(address)
        if keys is None:
            wanted = self._domain(address)
        else:
            wanted = [self._key(address, key) for key in self._as_keys(address, keys)]
        return {key: self._row(address, key) for key in wanted}

    @staticmethod
    def _as_keys(address, keys) -> list:
        """One key or many, normalized to a list without touching values."""
        if isinstance(keys, (str, bytes)):
            return [keys]
        if isinstance(keys, tuple) and address in (
            'layers',
            'node_layers',
            'edge_slices',
            'elementary_layers',
        ):
            return [keys]
        if isinstance(keys, RowSelection):
            return list(keys.keys)
        if hasattr(keys, 'ids') and not isinstance(keys, Mapping):
            return list(keys.ids)
        try:
            return list(keys)
        except TypeError:
            return [keys]

    # -- schema ------------------------------------------------------------------

    def _fields(self, address) -> list[str]:
        """The value fields an address holds now, in first-seen order."""
        g = self._G
        if address == 'nodes':
            return g._attr_store.node_column_names()
        if address == 'edges':
            return g._attr_store.edge_column_names()
        level = getattr(g._contextual, LEVEL_OF[address])
        hidden = _declared(address)
        seen: dict = {}
        for attrs in level.values():
            for name in attrs:
                if name not in hidden:
                    seen.setdefault(name, None)
        return list(seen)

    def _field_types(self, address) -> dict:
        g = self._G
        if address == 'nodes':
            return {
                name: _column_type(column) for name, column in g._attr_store.node_columns.items()
            }
        if address == 'edges':
            return {
                name: _column_type(column) for name, column in g._attr_store.edge_columns.items()
            }
        level = getattr(g._contextual, LEVEL_OF[address])
        hidden = _declared(address)
        types: dict = {}
        for attrs in level.values():
            for name, value in attrs.items():
                if _is_null(value) or name in hidden:
                    continue
                found = _type_name(value)
                held = types.get(name)
                types[name] = found if held is None or held == found else 'object'
        return types

    def schema(self, address, *, derived: bool = False) -> Schema:
        """Report what an address is keyed by and what it holds, cheaply.

        No frame is built and no backing is scanned. The value fields are the
        names the store holds; their types are read off the column dtype or
        from the values present.
        """
        self._address(address)
        keys = tuple((name, _SCHEMA_TYPE_OF_KEY[name]) for name in KEY_COLUMNS[address])
        types = self._field_types(address)
        fields = [(name, types.get(name, 'object')) for name in self._fields(address)]
        if derived:
            from . import _tables

            for name, kind in _tables.derived_columns(self._G, address):
                if name not in dict(keys) and name not in dict(fields):
                    fields.append((name, kind))
        return Schema(address, keys, tuple(fields), len(self._domain(address)), derived)

    # -- selecting rows -------------------------------------------------------------

    def _orders(self, address) -> dict:
        """The ordered aspects a layer-keyed address compares by, by field name."""
        g = self._G
        if address not in ('layers', 'node_layers') or g._aspects == ('_',):
            return {}
        orders = {}
        for name in g._aspects:
            aspect = g.layers.aspect(name)
            if aspect.ordered:
                orders[name] = aspect
        return orders

    def _query_fields(self, address) -> set:
        fields = set(self._fields(address)) | set(KEY_COLUMNS[address])
        g = self._G
        if address in ('layers', 'node_layers') and g._aspects != ('_',):
            fields.update(g._aspects)
        return fields

    def _query_row(self, address, key, row: dict) -> dict:
        """One row with its key columns spread, for the engine to read."""
        out = dict(row)
        names = KEY_COLUMNS[address]
        parts = (key,) if len(names) == 1 else tuple(key)
        out.update(zip(names, parts, strict=True))
        if address in ('layers', 'node_layers') and self._G._aspects != ('_',):
            coordinate = key if address == 'layers' else key[1]
            out.update(zip(self._G._aspects, coordinate, strict=False))
        return out

    def _rows_where(self, address, conditions) -> list:
        """The keys of this reader's domain whose rows satisfy the conditions."""
        domain = self._domain(address)
        if not conditions:
            return list(domain)
        pairs = ((key, self._query_row(address, key, self._row(address, key))) for key in domain)
        return rows_matching(pairs, conditions, orders=self._orders(address))

    def select(self, address, conditions=None, /, **keywords) -> RowSelection:
        """Select the rows of one address that satisfy every condition.

        The conditions are the ones ``G.N.select`` takes, over the address's
        key columns, its stored fields, and — for the layer-keyed addresses —
        the declared aspects, compared by their order when ordered. Keys are
        kept as the address keys them; see :meth:`RowSelection.project`.
        """
        self._address(address)
        parsed = parse_conditions(conditions, fields=self._query_fields(address), **keywords)
        g = self._G
        if address in ('layers', 'node_layers') and g._aspects != ('_',):
            for field, operator, wanted in parsed:
                if field in g._aspects and operator in ('lt', 'lte', 'gt', 'gte'):
                    # Compared by declared order; a categorical aspect has none.
                    g.layers.aspect(field).index(wanted)
        found = RowSelection(self, address, Leaf('where', parsed, self))
        # Resolve once now, so a value that cannot be compared is reported at
        # the call rather than at the first read.
        _ = found.keys
        return found

    def project_keys(self, rows: RowSelection, target: str) -> list:
        """The ids of ``target`` named by the rows, deduplicated in this reader's order."""
        return project_keys(self, rows, target)

    # -- writing ---------------------------------------------------------------------

    def _reserved(self, address, attrs: dict) -> None:
        bad = sorted(name for name in attrs if name in _RESERVED[address])
        if bad:
            raise ValueError(
                f'{address} attributes use reserved key(s): {bad!r}. '
                f'These names are part of the structural / dispatch contract; '
                f'rename your attribute(s) to use a different key.'
            )

    def _normalize_updates(self, address, updates, *, layer=None, key=None) -> dict:
        """Check every row of a batch and return ``{normalized key: {field: value}}``.

        Nothing is written here. A field whose value is null means "remove".
        """
        if (layer is not None or key is not None) and address != 'node_layers':
            raise TypeError('layer= and key= are shorthands for node_layers only')
        items = list(updates.items()) if isinstance(updates, Mapping) else list(updates)
        if address == 'node_layers':
            # Shape every row into a full placement before any key is compared,
            # and keep them as a list: a mapping built here would fold two rows
            # for one placement into one, and the repeat could not be seen.
            shaped = []
            for holder, value in items:
                if not (
                    isinstance(holder, tuple) and len(holder) == 2 and isinstance(holder[0], str)
                ):
                    if layer is None:
                        raise ValueError(f'{holder!r} is a bare node id, so this call needs layer=')
                    holder = (holder, layer)
                if not isinstance(value, Mapping):
                    if key is None:
                        raise ValueError(
                            f'the value for {holder!r} is a scalar, so this call needs key='
                        )
                    value = {key: value}
                shaped.append((holder, value))
            items = shaped
        normalized: dict = {}
        for raw_key, attrs in items:
            found = self._key(address, raw_key)
            if found in normalized:
                raise ValueError(f'duplicate key {found!r} in one batch (given as {raw_key!r})')
            if not isinstance(attrs, Mapping):
                raise TypeError(
                    f'each row is a mapping of attribute names to values; got '
                    f'{type(attrs).__name__} for {raw_key!r}'
                )
            attrs = dict(attrs)
            for name in attrs:
                if not isinstance(name, str):
                    raise TypeError(f'attribute names are strings, not {type(name).__name__}')
            self._reserved(address, attrs)
            if address == 'edge_slices' and 'weight' in attrs and not _is_null(attrs['weight']):
                try:
                    attrs['weight'] = float(attrs['weight'])
                except (TypeError, ValueError):
                    raise TypeError(
                        f'an edge-slice weight is a number, not {attrs["weight"]!r}'
                    ) from None
            normalized[found] = attrs
        return normalized

    def _preflight_composite_keys(self, updates: dict) -> dict:
        """Compute the composite-key index the batch would leave, or raise.

        The batch is judged by its final state, not row by row: every node the
        batch names gives up its old key first, then every new key is claimed.
        A key that two nodes would end up sharing, or that a node outside the
        batch still owns, is a collision.
        """
        g = self._G
        index = dict(g._node_key_index)
        new_keys = {}
        for node_id, attrs in updates.items():
            merged = g._attr_store.node_attrs(node_id)
            for name, value in attrs.items():
                if _is_null(value):
                    merged.pop(name, None)
                else:
                    merged[name] = value
            old_key = g._current_key_of_node(node_id)
            if old_key is not None and index.get(old_key) == node_id:
                del index[old_key]
            new_keys[node_id] = g._build_key_from_attrs(merged)
        for node_id, new_key in new_keys.items():
            if new_key is None:
                continue
            owner = index.get(new_key)
            if owner is not None and owner != node_id:
                raise ValueError(
                    f'Composite key collision on {g._node_key_fields}: {new_key} owned by {owner}'
                )
            index[new_key] = node_id
        return index

    def _restore_rows(self, address, snapshot: dict) -> None:
        """Put rows back exactly as ``snapshot`` holds them (the transaction's restorer)."""
        g = self._G
        for key, old in snapshot.items():
            if address in _GENERIC:
                current = self._row(address, key)
                clean = {name: None for name in current if name not in old}
                clean.update(old)
                if address == 'nodes':
                    g._attr_store.set_node_attrs(key, clean)
                else:
                    g._attr_store.set_edge_attrs(key, clean)
            else:
                level = LEVEL_OF[address]
                held = getattr(g._contextual, level)
                hidden = _declared(address)
                kept = {n: v for n, v in (held.get(key) or {}).items() if n in hidden}
                restored = {**kept, **old}
                if restored:
                    held[key] = restored
                else:
                    held.pop(key, None)
                g._contextual.touch(level)

    def _commit(self, address, updates: dict) -> None:
        """Write validated rows and fire the policies they touch. Called inside a transaction."""
        g = self._G
        if address == 'nodes':
            store = g._attr_store
            for key, attrs in updates.items():
                store.set_node_attrs(key, attrs)
            watched = g._variables_watched_by_nodes()
            if watched:
                affected: dict = {}
                for node_id, attrs in updates.items():
                    if watched.intersection(attrs):
                        for edge_id in g._incident_flexible_edges(node_id):
                            affected.setdefault(edge_id, None)
                for edge_id in affected:
                    g._apply_flexible_direction(edge_id)
            return
        if address == 'edges':
            store = g._attr_store
            for key, attrs in updates.items():
                store.set_edge_attrs(key, attrs)
            policies = g.edge_direction_policy
            for edge_id, attrs in updates.items():
                policy = policies.get(edge_id)
                if policy and policy.get('scope', 'edge') == 'edge' and policy['var'] in attrs:
                    g._apply_flexible_direction(edge_id)
            return
        level = LEVEL_OF[address]
        held = getattr(g._contextual, level)
        for key, attrs in updates.items():
            row = held.get(key)
            if row is None:
                row = {}
            for name, value in attrs.items():
                if _is_null(value):
                    row.pop(name, None)
                else:
                    row[name] = value
            if row:
                held[key] = row
            else:
                held.pop(key, None)
        g._contextual.touch(level)
        if address == 'elementary_layers':
            g._layer_table_passthrough = None

    def _read_schema(self, address):
        """What identifies the stored fields of a generic address, or ``None``."""
        if address == 'nodes':
            return self._G._attr_store.column_schema(NODE_AXIS)
        if address == 'edges':
            return self._G._attr_store.column_schema(EDGE_AXIS)
        return None

    def _restore_schema(self, address, schema) -> None:
        if address == 'nodes':
            self._G._attr_store.restore_column_schema(NODE_AXIS, schema)
        elif address == 'edges':
            self._G._attr_store.restore_column_schema(EDGE_AXIS, schema)

    def _transaction(self, address, keys, *, whole_axis: bool = False) -> AttributeTransaction:
        return AttributeTransaction(
            self._G,
            address,
            keys,
            read_row=self._row,
            restore_rows=self._restore_rows,
            read_schema=self._read_schema,
            restore_schema=self._restore_schema,
            whole_axis=whole_axis,
        )

    def update(self, address, updates, /, *, layer=None, key=None) -> int:
        """Merge attribute rows at one address, all of them or none.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        updates : Mapping[key, Mapping[str, Any]] | Iterable[tuple[key, Mapping]]
            The rows to merge. A field set to ``None`` (or NaN) is removed.
        layer : tuple, optional
            ``node_layers`` only: the layer of every bare node id in ``updates``.
        key : str, optional
            ``node_layers`` only: the attribute name of every scalar value.

        Returns
        -------
        int
            The number of rows written.

        Raises
        ------
        KeyError
            For a key the address does not hold. Nothing is written.
        ValueError
            For a reserved field, a duplicate key or a composite-key collision.
        TypeError
            For a row that is not a mapping or a value of the wrong type.

        Notes
        -----
        Everything is checked before the first cell changes, and a failure
        inside the commit — a flexible-direction policy raising after it has
        re-oriented an edge, for instance — rolls back the rows, the
        composite-key index and the rewritten incidence columns. See
        :mod:`annnet.core._transaction`.
        """
        self._writable()
        self._address(address)
        normalized = self._normalize_updates(address, updates, layer=layer, key=key)
        if not normalized:
            return 0
        self._apply(address, normalized)
        return len(normalized)

    def _apply(self, address, normalized: dict, *, prune=()) -> None:
        """Commit validated rows inside one transaction.

        ``prune`` names fields whose generic column is dropped when the write has
        left it empty, so a removed field is gone from the schema and not only
        from the rows. It happens inside the transaction: a failure restores it.
        """
        g = self._G
        index = None
        if address == 'nodes' and g._node_key_enabled():
            index = self._preflight_composite_keys(normalized)
        with self._transaction(address, normalized):
            self._commit(address, normalized)
            if index is not None:
                g._node_key_index = index
            if prune and address in _GENERIC:
                axis = NODE_AXIS if address == 'nodes' else EDGE_AXIS
                g._attr_store.drop_empty_columns(axis, prune)

    def replace(self, address, table) -> None:
        """Replace the attributes of a whole address with the rows of a table.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        table : DataFrame-like
            A table in any supported backend with the address's key columns and
            one column per attribute. Null cells mean "no such attribute".

        Notes
        -----
        A key the table omits loses its attributes and keeps its existence: the
        nodes, edges, slices and layers themselves are never added or removed
        by a table. The table is validated whole before anything changes, and
        the old attributes are restored if the write fails. Use :meth:`update`
        to change some fields of some rows.

        Raises
        ------
        ValueError
            For a missing key column, a null or duplicate key or a reserved
            field.
        KeyError
            For a key the address does not hold.
        """
        self._writable()
        self._replace_table(address, table)

    def delete(self, address, keys=None, names=None) -> int:
        """Remove attributes, never the structure that carries them.

        ================  ======================================================
        ``keys``          ``names``
        ================  ======================================================
        given             omitted: clear every attribute of those rows
        omitted           given: drop those fields from every row of the address
        given             given: drop those fields from those rows
        omitted           omitted: :class:`ValueError`, never "everything"
        ================  ======================================================

        An explicit empty ``keys`` or ``names`` selects nothing and does nothing.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        keys : key | Iterable[key] | RowSelection, optional
            The rows, keyed as the address keys them.
        names : str | Iterable[str], optional
            The attribute fields.

        Returns
        -------
        int
            The number of rows that lost at least one attribute.

        Raises
        ------
        ValueError
            When neither ``keys`` nor ``names`` is given, or a name is reserved.
        KeyError
            For a key the address does not hold, or a name the address has no
            such attribute for.
        """
        self._writable()
        self._address(address)
        if keys is None and names is None:
            raise ValueError(
                f'delete({address!r}) with neither keys nor names would remove every '
                f'attribute; name the rows with keys=... or the fields with names=...'
            )
        wanted = None
        if names is not None:
            listed = [names] if isinstance(names, str) else list(names)
            for name in listed:
                if not isinstance(name, str):
                    raise TypeError(f'attribute names are strings, not {type(name).__name__}')
            wanted = list(dict.fromkeys(listed))
            self._reserved(address, dict.fromkeys(wanted))
            known = set(self._fields(address))
            unknown = [name for name in wanted if name not in known]
            if unknown:
                raise KeyError(
                    f'{address} holds no attribute {unknown!r}; it holds {sorted(known)!r}'
                )
        if keys is None:
            targets = self._domain(address)
        else:
            targets = list(
                dict.fromkeys(self._key(address, key) for key in self._as_keys(address, keys))
            )
        if wanted is not None and not wanted or not targets:
            return 0
        changes: dict = {}
        for key in targets:
            row = self._row(address, key)
            gone = list(row) if wanted is None else [name for name in wanted if name in row]
            if gone:
                changes[key] = dict.fromkeys(gone)
        if changes:
            prune = (
                wanted if wanted is not None else sorted({n for c in changes.values() for n in c})
            )
            self._apply(address, changes, prune=prune)
        return len(changes)

    def from_frame(self, address, frame) -> RowSelection:
        """Select the rows of a frame, by the keys it names.

        Read a table, filter it with the dataframe library it came in, and pass
        what is left here to get a selection of the graph's own keys::

            table = G.attrs.table('nodes')
            kept = table.filter(pl.col('score') > 0.5)
            chosen = G.attrs.from_frame('nodes', kept)

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        frame : DataFrame-like
            Any supported backend. Only the address's key columns are read.

        Returns
        -------
        RowSelection
            A selection of exactly those keys, in the address's own order and
            without repeats. It is fixed: it is the identities the frame named,
            not the condition that produced the frame, so a key added later is
            not picked up, and a key that is removed and added again is the same
            key. An empty frame gives an empty selection.

        Raises
        ------
        ValueError
            For a missing key column or a null key.
        KeyError
            For a key the address does not hold (or, on a view, one outside it).
        """
        self._address(address)
        key_columns = KEY_COLUMNS[address]
        columns = list(dataframe_columns(frame))
        missing = [name for name in key_columns if name not in columns]
        if missing:
            raise ValueError(
                f'a {address} frame needs the key column(s) {list(key_columns)!r}; missing {missing!r}'
            )
        found = set()
        for row in dataframe_to_rows(frame):
            parts = tuple(row[name] for name in key_columns)
            if any(part is None for part in parts):
                raise ValueError(f'{address} key columns must not be null; got {parts!r}')
            parts = tuple(
                _coordinate(part) if name == 'layer' else part
                for name, part in zip(key_columns, parts, strict=True)
            )
            found.add(self._key(address, parts[0] if len(parts) == 1 else parts))
        return RowSelection(self, address, Leaf('ids', frozenset(found), self))

    def _replace_table(self, address, frame) -> None:
        """Replace every row of one address with the rows of a table.

        Keys the table omits lose their attributes; the elements stay. The
        whole table is validated before anything changes, and the old rows are
        restored if the commit fails.
        """
        self._address(address)
        keys = KEY_COLUMNS[address]
        columns = list(dataframe_columns(frame))
        if (
            address == 'elementary_layers'
            and 'layer_id' in columns
            and not set(keys) <= set(columns)
        ):
            key_columns: tuple = ('layer_id',)
        else:
            key_columns = keys
        missing = [name for name in key_columns if name not in columns]
        if missing:
            raise ValueError(
                f'a {address} table needs the key column(s) {list(keys)!r}; missing {missing!r}'
            )
        updates: dict = {}
        for row in dataframe_to_rows(frame):
            row = dict(row)
            parts = tuple(row.pop(name) for name in key_columns)
            for name in keys:
                row.pop(name, None)
            if address in ('layers', 'node_layers', 'elementary_layers'):
                # A display id beside the structured key is not an attribute.
                row.pop('layer_id', None)
                row.pop('coordinate_id', None)
            if any(part is None for part in parts):
                raise ValueError(f'{address} table keys must not be null; got {parts!r}')
            if 'layer' in key_columns:
                parts = tuple(
                    _coordinate(part) if name == 'layer' else part
                    for name, part in zip(key_columns, parts, strict=True)
                )
            raw = parts[0] if len(parts) == 1 else parts
            found = self._key(address, raw)
            if found in updates:
                raise ValueError(f'duplicate table key {found!r}')
            updates[found] = dict(row.items())
        for attrs in updates.values():
            self._reserved(address, attrs)
        g = self._G
        domain = self._domain(address)
        # Every field of every current row goes, then the table's rows land.
        replacement: dict = {}
        for found in domain:
            replacement[found] = dict.fromkeys(self._row(address, found))
        for found, attrs in updates.items():
            replacement.setdefault(found, {}).update(attrs)
        if address == 'edge_slices':
            for attrs in replacement.values():
                if 'weight' in attrs and not _is_null(attrs['weight']):
                    attrs['weight'] = float(attrs['weight'])
        replacement = {
            found: {name: value for name, value in attrs.items() if not _is_null(value)}
            for found, attrs in replacement.items()
            if any(not _is_null(value) for value in attrs.values())
        }
        index = None
        if address == 'nodes' and g._node_key_enabled():
            index = self._preflight_composite_keys(replacement)
        with self._transaction(address, domain, whole_axis=True):
            if address == 'nodes':
                g._attr_store.drop_node_columns()
            elif address == 'edges':
                g._attr_store.drop_edge_columns()
            else:
                level = LEVEL_OF[address]
                held = getattr(g._contextual, level)
                hidden = _declared(address)
                # A declaration the level keeps beside the attributes is not
                # part of the table, so a table replacement leaves it alone.
                declarations = {
                    key: {n: v for n, v in attrs.items() if n in hidden}
                    for key, attrs in held.items()
                    if any(n in hidden for n in attrs)
                }
                g._contextual.clear_level(level)
                for key, attrs in declarations.items():
                    held[key] = dict(attrs)
            self._commit(address, replacement)
            if index is not None:
                g._node_key_index = index
        if address == 'elementary_layers':
            g._layer_table_passthrough = None

    # -- tables --------------------------------------------------------------------

    def _stored_table(self, address, backend, limit=None):
        """The stored table of one address in one backend, scoped.

        A ``limit`` reaches the walk over the keys, so a preview costs its rows.
        """
        g = self._G
        if (
            self._scope is None
            and address in _GENERIC
            and limit is None
            and not any(
                _holds_containers(c)
                for c in (
                    g._attr_store.node_columns if address == 'nodes' else g._attr_store.edge_columns
                ).values()
            )
        ):
            # Built in the backend asked for rather than converted into it: a
            # conversion between backends does not keep every null a null.
            store = g._attr_store
            return store.obs(backend=backend) if address == 'nodes' else store.var(backend=backend)
        keys = KEY_COLUMNS[address]
        rows = []
        domain = self._domain(address)
        if limit is not None:
            domain = itertools.islice(domain, limit)
        for key in domain:
            attrs = self._row(address, key)
            row = dict(zip(keys, (key,) if len(keys) == 1 else key, strict=True))
            if 'layer' in row:
                row['layer'] = list(row['layer'])
            row.update(attrs)
            rows.append(row)
        if not rows:
            schema = {name: ('list_text' if name == 'layer' else 'text') for name in keys}
            return empty_dataframe(schema, backend=backend)
        return dataframe_from_rows(rows, backend=backend)

    def table(
        self,
        address,
        *,
        derived: bool = False,
        backend: str | None = None,
        columns=None,
        limit: int | None = None,
        layout: str | None = None,
        **query,
    ):
        """Render one address as a table, stored or derived.

        Parameters
        ----------
        address : str
            One of :data:`ADDRESSES`.
        derived : bool, default False
            ``True`` asks for everything the graph can say about the subject —
            structural columns, joins and filters — through ``**query``.
            ``False`` is the stored attributes alone.
        backend : str, optional
            One of the dataframe backends; defaults to :attr:`backend`.
        columns : Iterable[str], optional
            Keep only these value columns. The key columns are always kept.
        limit : int, optional
            Keep only the first ``limit`` rows, in the address's order. A
            preview, never a selection: nothing about the graph or the view
            changes.
        layout : {"edges", "incidences"}, optional
            ``edges`` (the default) gives one row per edge; ``incidences`` gives
            one row per edge endpoint with the participant's identity, layer,
            role and coefficient. Edges only, derived only.
        **query
            Derived-table arguments: ``slice=`` (join one slice's overrides),
            ``in_slice=`` (keep members of one slice), ``layer=``,
            ``include_hyper=``, ``include_binary=``, ``include_directed=``,
            ``include_weight=``, ``resolved_weight=`` for edges. See
            :mod:`annnet.core._tables`.

        Returns
        -------
        DataFrame-like
            A caller-owned frame; editing it changes nothing.
        """
        self._address(address)
        backend = backend or self.backend
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError(f'limit is a non-negative integer, not {limit!r}')
        if not derived:
            if query or layout is not None:
                raise TypeError('query arguments and layout= need derived=True')
            frame = self._stored_table(address, backend, limit)
        else:
            from . import _tables

            domain = (
                self._domain(address) if address in ('edge_slices', 'elementary_layers') else None
            )
            frame = _tables.derived_table(
                self._G,
                address,
                scope=self._scope,
                domain=domain,
                layout=layout,
                backend=backend,
                limit=limit,
                **query,
            )
        return _finish_table(
            frame,
            KEY_COLUMNS[address],
            columns=columns,
            limit=limit,
            backend=backend,
            layout=layout,
        )

    # -- audit --------------------------------------------------------------------

    def audit(self) -> dict:
        """Report attribute rows whose structural keys no longer exist.

        The generic tables are derived from slot-indexed columns, so they can
        hold no extra row and miss none; what an audit still finds is a
        contextual row naming a slice, edge, node, layer or placement the
        graph does not hold.
        """
        g = self._G
        out: dict = {}
        node_ids = set(_structure.node_ids(g))
        edge_ids = set(_structure.edge_ids(g)) | set(g._store.live_edge_ids())
        placements = set(_structure.node_keys(g))
        contextual = g._contextual
        out['invalid_slice_rows'] = [key for key in contextual.slice_attrs if key not in g._slices]
        out['invalid_edge_slice_rows'] = [
            key
            for key in contextual.edge_slice_attrs
            if key[0] not in g._slices or key[1] not in edge_ids
        ]
        out['invalid_node_layer_rows'] = [
            key for key in contextual.node_layer_attrs if key not in placements
        ]
        out['invalid_aspect_rows'] = [
            key for key in contextual.aspect_attrs if key not in g._aspects
        ]
        valid_layers = {tuple(aa) for aa in g.layers._all_layers}
        out['invalid_layer_rows'] = [
            key for key in contextual.layer_attrs if key not in valid_layers
        ]
        out['invalid_elementary_rows'] = [
            key
            for key in contextual.elementary_attrs
            if not (
                isinstance(key, tuple)
                and key[0] in g._aspects
                and key[1] in g._layers.get(key[0], ())
            )
        ]
        # The generic tables are derived from slot-indexed columns, so these four
        # are always empty; they are kept so that an older reader of the report
        # finds the keys it knows.
        node_attr_ids = set(g._attr_store.node_ids())
        edge_attr_ids = set(g._attr_store.edge_ids())
        out['extra_node_rows'] = sorted(node_attr_ids - node_ids)
        out['extra_edge_rows'] = sorted(edge_attr_ids - edge_ids)
        out['missing_node_rows'] = sorted(node_ids - node_attr_ids)
        out['missing_edge_rows'] = sorted(set(_structure.edge_ids(g)) - edge_attr_ids)
        return out


class ScopedAttrs(Attrs):
    """``V.attrs``: the eight addresses restricted to a view, read-only.

    ``owner`` is any object answering ``_selection_context()`` with
    ``(graph, resolved)``; the scope is re-read on every access, so a live
    view never reads through a stale membership.
    """

    __slots__ = ('_owner',)

    def __init__(self, owner):
        graph, _resolved = owner._selection_context()
        Attrs.__init__(self, graph)
        self._owner = owner

    @property
    def _scope(self):
        return self._owner._selection_context()[1]

    def _writable(self, what: str = 'a view') -> None:
        raise refuse_write(what)

    def _selection_owner(self):
        return self._owner

    def _is_live(self) -> bool:
        probe = getattr(self._owner, '_is_live', None)
        return bool(probe()) if callable(probe) else False

    def _root_reader(self):
        return self._G.attrs


def _finish_table(frame, keys, *, columns, limit, backend, layout):
    """Apply ``columns=`` and ``limit=`` to a rendered frame."""
    if columns is None and limit is None:
        return clone_dataframe(frame)
    rows = dataframe_to_rows(frame)
    if limit is not None:
        rows = rows[:limit]
    held = list(dataframe_columns(frame))
    if columns is not None:
        wanted = [columns] if isinstance(columns, str) else list(columns)
        unknown = [name for name in wanted if name not in held]
        if unknown:
            raise KeyError(f'unknown column(s) {unknown!r}; this table has {held!r}')
        key_names = [
            name
            for name in held
            if name in keys or (layout == 'incidences' and name in ('edge_id', 'position'))
        ]
        keep = key_names + [name for name in wanted if name not in key_names]
        rows = [{name: row.get(name) for name in keep} for row in rows]
        held = keep
    if not rows:
        return empty_dataframe(dict.fromkeys(held, 'text'), backend=backend)
    return dataframe_from_rows(rows, backend=backend)


def _install_table_properties() -> None:
    """Give ``Attrs`` one property per address: read the table, refuse assignment."""
    for address in ADDRESSES:

        def read(self, _address=address):
            return self.table(_address)

        def write(self, frame, _address=address):
            self._writable()
            raise TypeError(
                f'G.attrs.{_address} is read-only; replace the table with '
                f'G.attrs.replace({_address!r}, table)'
            )

        setattr(
            Attrs,
            address,
            property(
                read, write, doc=f'Attributes keyed by {address.replace("_", " ")}, as a table.'
            ),
        )


_install_table_properties()


__all__ = [
    'ADDRESSES',
    'KEY_COLUMNS',
    'LEVEL_OF',
    'Attrs',
    'ReadOnlyViewError',
    'Schema',
    'ScopedAttrs',
    'refuse_write',
]
