"""The one store of contextual attributes.

A generic attribute belongs to one element and lives in a slot-indexed column,
which :mod:`annnet.core._attrs` owns. A **contextual** attribute belongs to a
*pair* — one edge inside one slice, one node inside one layer, one label inside
one aspect. Almost no pair carries a value, so a dense column per pair would
waste nearly every cell, and these stores stay keyed by the pair instead.

**Canonical contextual state is a plain dict, never a dataframe.** Three of these
levels used to be stored as backend dataframes on the graph, which had two costs.

*It made the graph's own state depend on an optional dependency.* A stored
``polars.DataFrame`` means the type of a canonical field, its null handling and
its dtype promotion are decided by which library happens to be installed. Nothing
about a graph should change because a different table engine is available.

*It made a write cost the size of the table.* Every single attribute write went
through an upsert that filtered the whole frame and stacked one row back on, so
the per-write cost grew with the table and the total was quadratic — 3 200 writes
took over 17 seconds, against a flat 0.0025 ms per write on the generic axes. A
dict write is one hash insert and does not care how much is already there.

A dataframe is what a *reader* gets, built on demand in whatever backend the
caller asks for. That is what makes the backend a rendering choice rather than a
property of the graph, and it is why Narwhals appears only at the boundary.
"""

from __future__ import annotations

from typing import Any

# The six levels, each named by the pair it is keyed on. Declared once so that a
# copy, a clone and a serializer cannot each remember a different list.
LEVELS = (
    'slice_attrs',
    'edge_slice_attrs',
    'node_layer_attrs',
    'aspect_attrs',
    'layer_attrs',
    'elementary_attrs',
)


class ContextualStore:
    """Every contextual attribute of one graph, keyed by its own pair.

    One owner, one shape. Before this existed the six levels were split across
    three mechanisms and two owners — dataframes on the graph for the slice
    family, dicts on the layer accessor for the layer family — with nothing to
    explain which a given level got.
    """

    # The level names, reachable from any store. A caller outside the core needs
    # them to walk every level — serializing one, copying one — and reaching for
    # the module constant would cross a boundary to read a name the object can
    # answer for itself.
    levels = LEVELS

    __slots__ = (*LEVELS, 'version')

    def __init__(self) -> None:
        # Rises on every write. A materialized table records the value it was
        # built at, so a read after a read is free and a read after a write
        # rebuilds. Counting entries would not do: updating a value in place
        # changes no count.
        self.version = 0
        # slice_id            -> attrs
        self.slice_attrs: dict[str, dict] = {}
        # (slice_id, edge_id) -> attrs
        self.edge_slice_attrs: dict[tuple[str, str], dict] = {}
        # (node_id, layer)    -> attrs
        self.node_layer_attrs: dict[tuple[Any, tuple], dict] = {}
        # aspect              -> attrs
        self.aspect_attrs: dict[str, dict] = {}
        # layer coordinate    -> attrs
        self.layer_attrs: dict[tuple, dict] = {}
        # (aspect, label)     -> attrs
        self.elementary_attrs: dict[tuple[str, str], dict] = {}

    # -- lifecycle --------------------------------------------------------

    def copy(self) -> ContextualStore:
        """Return a store holding the same values and sharing no dict with this one."""
        other = ContextualStore()
        other.version = self.version
        for level in LEVELS:
            getattr(other, level).update(
                {key: dict(value) for key, value in getattr(self, level).items()}
            )
        return other

    def clear(self) -> None:
        """Forget every level."""
        for level in LEVELS:
            getattr(self, level).clear()
        self.version += 1

    def is_empty(self) -> bool:
        """Whether any level carries a value."""
        return not any(getattr(self, level) for level in LEVELS)

    # -- forgetting one element ------------------------------------------

    def forget_edge(self, edge_id: str) -> None:
        """Drop every pair that names one edge."""
        for key in [key for key in self.edge_slice_attrs if key[1] == edge_id]:
            del self.edge_slice_attrs[key]
        self.version += 1

    def forget_node(self, node_id) -> None:
        """Drop every pair that names one node.

        Membership names bare ids, so an entity key is reduced to its id the way
        the rest of the contextual API does.
        """
        bare = node_id[0] if isinstance(node_id, tuple) else node_id
        for key in [
            key
            for key in self.node_layer_attrs
            if (key[0][0] if isinstance(key[0], tuple) else key[0]) == bare
        ]:
            del self.node_layer_attrs[key]
        self.version += 1

    def forget_slice(self, slice_id: str) -> None:
        """Drop a slice and every edge pair inside it."""
        self.slice_attrs.pop(slice_id, None)
        for key in [key for key in self.edge_slice_attrs if key[0] == slice_id]:
            del self.edge_slice_attrs[key]
        self.version += 1

    # -- one level, one pair ---------------------------------------------
    #
    # Every write goes through here, so that the version rises exactly once per
    # change and nothing outside has to remember to bump it.

    def set(self, level_name: str, key, attrs: dict) -> None:
        """Merge attributes into one pair of one level."""
        if not attrs:
            return
        getattr(self, level_name).setdefault(key, {}).update(attrs)
        self.version += 1

    def get(self, level_name: str, key) -> dict:
        """Return a copy of what one pair carries, or an empty dict."""
        return dict(getattr(self, level_name).get(key, {}))

    def replace(self, level_name: str, contents) -> None:
        """Replace a whole level with a mapping."""
        level = getattr(self, level_name)
        level.clear()
        level.update({key: dict(value) for key, value in (contents or {}).items()})
        self.version += 1

    def touch(self) -> None:
        """Record that a level was changed by something holding it directly."""
        self.version += 1


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------
#
# A reader asks for a table; the store holds dicts. These build the one from the
# other, in the backend the caller names, and they are the only place in the
# contextual path that knows a dataframe exists.


def rows_of(level: dict, key_columns) -> list[dict]:
    """Return one row per pair, with the pair spread across its key columns.

    Rows come out sorted by key so that two graphs holding the same values
    materialize the same table — a writer that hashes its output, and a test that
    compares two tables, both need that.
    """
    keys = (key_columns,) if isinstance(key_columns, str) else tuple(key_columns)
    rows = []
    for key, attrs in level.items():
        parts = (key,) if len(keys) == 1 else tuple(key)
        row = dict(zip(keys, parts, strict=False))
        row.update(attrs)
        rows.append(row)
    rows.sort(key=lambda row: tuple(str(row.get(name, '')) for name in keys))
    return rows


def install_rows(level: dict, rows, key_columns) -> None:
    """Fill one level from table rows, replacing whatever it held.

    The inverse of :func:`rows_of`, used by a loader, a copy and an adapter that
    hand over a table rather than a mapping.
    """
    keys = (key_columns,) if isinstance(key_columns, str) else tuple(key_columns)
    level.clear()
    for row in rows or ():
        row = dict(row)
        parts = tuple(row.pop(name, None) for name in keys)
        if any(part is None for part in parts):
            continue
        attrs = {name: value for name, value in row.items() if value is not None}
        level[parts[0] if len(keys) == 1 else parts] = attrs
