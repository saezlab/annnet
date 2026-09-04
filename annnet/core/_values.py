"""Where the value of one node inside one layer comes from.

A node-layer attribute is a value keyed by a pair — one node, one layer. The
contextual store holds those in a dict, which is the right shape when a person
typed them: a handful of layers, a handful of nodes, almost every pair carrying
nothing.

It is the wrong shape when the layers come from a measurement table. A dict entry
costs about 360 bytes and a few microseconds; twenty thousand nodes across a
thousand layers is two times ten to the seventh cells, which is minutes to write
and gigabytes to hold. The same numbers as a dense ``float32`` array are eighty
megabytes and a microsecond a read. The difference is not the implementation, it
is the shape.

So a graph may hold either, and this module is the seam. A reader asks a
*resolver* for cells; the resolver asks each backing in turn. Nothing above this
line knows which one answered, which is what lets one analysis run at both
scales.

Two ways to ask
---------------

:meth:`ValueResolver.get` answers one cell and always works.
:meth:`ValueResolver.block` answers a whole rectangle at once, and returns
``None`` when it cannot — when the cells span several backings, or when one of
them is the dict store. ``None`` is a signal to use the slow path, never a wrong
answer, and the two paths are pinned equal by test.
"""

from __future__ import annotations

from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import numpy as np

#: Returned by a backing that holds nothing for a cell, so that a stored ``None``
#: is told from an absent one.
MISSING = object()


class ValueBacking(Protocol):
    """One place node-layer values may live."""

    def names(self) -> set[str]:
        """The attribute names this backing can answer for."""
        ...

    def get(self, node_id: str, layer: tuple, name: str, default: Any = None) -> Any:
        """The value of one cell, or ``default`` when this backing has none."""
        ...

    def block(self, nodes: Sequence[str], layers: Sequence[tuple], name: str):
        """``(values, answered)`` for a rectangle, or ``None`` when unable."""
        ...

    def layers(self) -> set[tuple]:
        """The layers this backing holds a value in."""
        ...

    def nodes(self) -> set[str]:
        """The nodes this backing holds a value for."""
        ...


class ContextualValues:
    """The contextual store, read as a backing.

    Canonical for values a caller sets one at a time through
    :meth:`LayerAccessor.set_node_attrs`. A pair carrying nothing occupies
    nothing, which is what makes it right for a sparse, hand-written table and
    wrong for a dense one.
    """

    __slots__ = ('_level',)

    def __init__(self, level: dict) -> None:
        self._level = level

    def names(self) -> set[str]:
        return {name for attrs in self._level.values() for name in attrs}

    def get(self, node_id: str, layer: tuple, name: str, default: Any = None) -> Any:
        return self._level.get((node_id, tuple(layer)), {}).get(name, default)

    def block(self, nodes, layers, name):
        """Always ``None``: a dict has no rectangle to hand back.

        Declared rather than left off, so a reader can ask every backing the same
        question and read the refusal as an answer.
        """
        return None

    def layers(self) -> set[tuple]:
        return {key[1] for key in self._level}

    def nodes(self) -> set[str]:
        return {key[0] for key in self._level}

    def __repr__(self) -> str:
        return f'ContextualValues({len(self._level)} pairs)'


class MatrixValues:
    """An array of values, addressed by two index maps.

    One array per attribute name, laid out layer by node, plus the two maps that
    say which row is which layer and which column is which node. That is the
    shape a measurement table already has, so attaching one costs building the
    maps and nothing else — no cell is copied, and the array stays whatever it
    was.

    Parameters
    ----------
    arrays : dict[str, array-like]
        One two-dimensional array per attribute name, each ``len(layers)`` rows
        by ``len(nodes)`` columns.
    layers : Sequence[tuple]
        The layer each row stands for, in row order.
    nodes : Sequence[str]
        The node each column stands for, in column order.
    rows, columns : dict, optional
        Explicit index maps, when the defaults from ``layers`` and ``nodes`` are
        not what is wanted. An explicit column map lets two nodes share one
        column, which is what a join of many nodes onto one measured entity
        needs — and it needs it without copying the column, which is the whole
        reason the array is attached rather than unpacked.
    mask : array-like, optional
        A boolean array the same shape as the values. A cell it gates out holds
        no value whatever the array carries there.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, if one is not two-dimensional, if one is too
        small for the maps, or if ``mask`` does not match their shape.
    """

    __slots__ = ('_arrays', '_row_of', '_col_of', '_layers', '_nodes', '_mask')

    def __init__(
        self,
        arrays: dict[str, Any],
        layers: Sequence[tuple],
        nodes: Sequence[str],
        *,
        rows: dict | None = None,
        columns: dict | None = None,
        mask: Any = None,
    ) -> None:
        if not arrays:
            raise ValueError('attach at least one array; there is nothing to address otherwise')
        self._layers = [tuple(layer) for layer in layers]
        self._nodes = list(nodes)
        self._row_of = (
            {tuple(layer): index for layer, index in rows.items()}
            if rows is not None
            else {layer: index for index, layer in enumerate(self._layers)}
        )
        self._col_of = (
            dict(columns)
            if columns is not None
            else {node: index for index, node in enumerate(self._nodes)}
        )
        self._mask = mask
        height = 1 + max(self._row_of.values(), default=-1)
        width = 1 + max(self._col_of.values(), default=-1)
        self._arrays: dict[str, Any] = {}
        for name, array in arrays.items():
            shape = getattr(array, 'shape', None)
            if shape is None or len(shape) != 2:
                raise ValueError(f'{name!r} is not a two-dimensional array')
            if shape[0] < height or shape[1] < width:
                raise ValueError(
                    f'{name!r} has shape {tuple(shape)}, and the maps address {(height, width)}'
                )
            self._arrays[name] = array
        if mask is not None:
            one = next(iter(self._arrays.values()))
            mask_shape = getattr(mask, 'shape', None)
            if mask_shape is None or tuple(mask_shape)[:2] != tuple(one.shape)[:2]:
                raise ValueError('mask must have the same shape as the arrays it gates')

    def names(self) -> set[str]:
        return set(self._arrays)

    def get(self, node_id: str, layer: tuple, name: str, default: Any = None) -> Any:
        array = self._arrays.get(name)
        if array is None:
            return default
        row = self._row_of.get(tuple(layer))
        column = self._col_of.get(node_id)
        if row is None or column is None:
            return default
        # A gated-out cell holds no value, whatever the array happens to carry
        # there: the gate says the node-layer was not measured.
        if self._mask is not None and not bool(self._mask[row, column]):
            return default
        value = array[row, column]
        return value.item() if hasattr(value, 'item') else value

    def block(self, nodes: Sequence[str], layers: Sequence[tuple], name: str):
        """``(values, answered)`` for one rectangle, read straight off the array.

        This is what the array was attached for. The per-cell path costs a few
        microseconds a cell and dominates any read of a real measurement table;
        fancy-indexing the same cells costs one pass in C.

        Parameters
        ----------
        nodes : Sequence[str]
            The node of each column, in order.
        layers : Sequence[tuple]
            The layer of each row, in order.
        name : str
            The attribute.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray] | None
            The values, and a boolean array of which cells this backing actually
            answers for. ``None`` when it does not hold ``name`` at all.
        """
        array = self._arrays.get(name)
        if array is None:
            return None
        rows = np.fromiter(
            (self._row_of.get(tuple(layer), -1) for layer in layers), dtype=int, count=len(layers)
        )
        columns = np.fromiter(
            (self._col_of.get(node, -1) for node in nodes), dtype=int, count=len(nodes)
        )
        answered = (rows[:, None] >= 0) & (columns[None, :] >= 0)
        # An unmapped row or column indexes position zero and is then masked out,
        # which keeps the gather one operation instead of a loop with a branch.
        taken = np.asarray(array)[
            np.ix_(np.where(rows >= 0, rows, 0), np.where(columns >= 0, columns, 0))
        ]
        values = np.asarray(taken, dtype=float)
        if self._mask is not None:
            gate = np.asarray(self._mask)[
                np.ix_(np.where(rows >= 0, rows, 0), np.where(columns >= 0, columns, 0))
            ]
            answered = answered & np.asarray(gate, dtype=bool)
        return values, answered

    def layers(self) -> set[tuple]:
        return set(self._row_of)

    def nodes(self) -> set[str]:
        return set(self._col_of)

    def __repr__(self) -> str:
        return (
            f'MatrixValues(names={sorted(self._arrays)!r}, '
            f'{len(self._row_of)} layers x {len(self._col_of)} nodes)'
        )


class ValueResolver:
    """Every backing of one graph, asked in order.

    A later backing wins for a cell it can answer, so attaching a table shadows
    whatever the contextual store held for the same pair rather than blending
    with it. Two sources for one cell is a conflict, and blending would hide it.
    """

    __slots__ = ('_backings',)

    def __init__(self, backings: Sequence[Any]) -> None:
        self._backings = list(backings)

    def names(self) -> set[str]:
        found: set[str] = set()
        for backing in self._backings:
            found |= backing.names()
        return found

    def layers(self) -> set[tuple]:
        found: set[tuple] = set()
        for backing in self._backings:
            found |= backing.layers()
        return found

    def nodes(self) -> set[str]:
        found: set[str] = set()
        for backing in self._backings:
            found |= backing.nodes()
        return found

    def get(self, node_id: str, layer: tuple, name: str, default: Any = None) -> Any:
        for backing in reversed(self._backings):
            value = backing.get(node_id, layer, name, MISSING)
            if value is not MISSING:
                return value
        return default

    def block(
        self,
        nodes: Sequence[str],
        layers: Sequence[tuple],
        name: str,
        default: Any = np.nan,
    ):
        """One rectangle of values, or ``None`` when it cannot be read as one.

        Every backing holding ``name`` is asked for its rectangle and they are
        laid over each other in attachment order, so the same "later wins" rule
        :meth:`get` follows also holds here — including the second backing an
        aggregate lands in.

        ``None`` comes back when some holder has no rectangle to give, which the
        dict store never does. It means *ask cell by cell*, and it is never a
        wrong answer.

        Parameters
        ----------
        nodes, layers : Sequence
            The columns and rows of the rectangle, in order.
        name : str
            The attribute.
        default : Any, default ``numpy.nan``
            What a cell no backing answers for holds.

        Returns
        -------
        numpy.ndarray | None
            ``len(layers)`` by ``len(nodes)``.
        """
        holders = [backing for backing in self._backings if name in backing.names()]
        if not holders:
            return None
        out = np.full((len(layers), len(nodes)), default, dtype=float)
        for backing in holders:
            found = backing.block(nodes, layers, name)
            if found is None:
                return None
            values, answered = found
            out = np.where(answered, values, out)
        return out

    def cells(
        self,
        nodes: Sequence[str],
        layers: Sequence[tuple],
        names: Sequence[str],
        default: Any = np.nan,
    ) -> Iterator[tuple]:
        """Yield ``(node_id, layer, name, value)`` for every asked-for cell."""
        for layer in layers:
            for node_id in nodes:
                for name in names:
                    yield node_id, layer, name, self.get(node_id, layer, name, default)

    def __repr__(self) -> str:
        return f'ValueResolver({self._backings!r})'


@dataclass(frozen=True, slots=True)
class ValueMatrix:
    """One attribute, as an array and the two labels that index it.

    What a method wants when it is handed a graph. A frame of Python objects has
    to be unpacked before any arithmetic; this is the arithmetic's own shape,
    plus the labels needed to put an answer back.

    Attributes
    ----------
    values : numpy.ndarray
        ``len(layers)`` rows by ``len(nodes)`` columns.
    nodes : list[str]
        The node of each column, in order.
    layers : list[tuple]
        The layer of each row, in order.
    name : str
        The attribute read.
    """

    values: Any
    nodes: list
    layers: list
    name: str

    @property
    def shape(self) -> tuple:
        """The shape of :attr:`values`."""
        return tuple(self.values.shape)

    def __array__(self, dtype=None, copy=None):
        """Read as an array, so ``numpy.asarray(m)`` is the values."""
        array = np.asarray(self.values, dtype=dtype)
        return array.copy() if copy else array

    def __repr__(self) -> str:
        return f'ValueMatrix({self.name!r}, {len(self.layers)} layers x {len(self.nodes)} nodes)'
