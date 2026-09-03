"""A window over the layers of a graph, named on the aspects rather than sliced.

Selecting layers meant building the coordinates by hand — a comprehension over
``_all_layers`` with an index into the tuple, and the aspect's position written
out at the call site::

    window = [aa for aa in G.layers._all_layers if aa[0] in TIMES[:3]]

Three facts about the graph live in that line: which position ``time`` is in,
what its values are, and what order they come in. All three are the graph's, and
all three go wrong silently the first time an aspect is added.

:meth:`LayerAccessor.where` takes them back:

    G.layers.where(time__lte='12h')

The window itself is resolved off the aspect declaration, so it costs the number
of *layers* and not the size of the graph. What the window is then asked for —
its nodes, its edges, what crosses its boundary — costs one pass over the axis
being asked about, and no more than one.
"""

from __future__ import annotations

from typing import Any

from . import _structure
from ._records import as_endpoint

#: The comparison suffixes a predicate may carry. ``eq`` is what a bare
#: ``aspect=value`` means.
OPERATORS = ('eq', 'ne', 'in', 'not_in', 'lt', 'lte', 'gt', 'gte')

#: The ones that ask where a value sits rather than what it is. Only an ordered
#: aspect can answer these.
ORDERED_ONLY = ('lt', 'lte', 'gt', 'gte')


def parse_predicate(key: str, aspects) -> tuple[str, str]:
    """Split one ``aspect__op`` keyword into its aspect and its operator.

    Parameters
    ----------
    key : str
        ``"time"``, or ``"time__lte"``.
    aspects : Sequence[str]
        The declared aspects, for the error message and to keep an aspect whose
        own name contains ``"__"`` readable.

    Returns
    -------
    tuple[str, str]
        ``(aspect, operator)``.

    Raises
    ------
    KeyError
        If the aspect is not declared.
    ValueError
        If the operator is not one of :data:`OPERATORS`.
    """
    if key in aspects:
        return key, 'eq'
    name, _, operator = key.rpartition('__')
    if not name:
        raise KeyError(f'unknown aspect {key!r}; this graph declares {list(aspects)!r}')
    if name not in aspects:
        raise KeyError(f'unknown aspect {name!r}; this graph declares {list(aspects)!r}')
    if operator not in OPERATORS:
        raise ValueError(
            f'unknown operator {operator!r} in {key!r}; one of {list(OPERATORS)} is expected'
        )
    return name, operator


def satisfies(aspect, operator: str, value: Any, wanted: Any) -> bool:
    """Whether one layer's value for one aspect satisfies one predicate.

    Parameters
    ----------
    aspect : Aspect
        The aspect the value belongs to; consulted for order.
    operator : str
        One of :data:`OPERATORS`.
    value : Any
        What this layer holds for the aspect.
    wanted : Any
        What the predicate asked for.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If a comparison in :data:`ORDERED_ONLY` is asked of a categorical aspect.
    """
    if operator == 'eq':
        return value == wanted
    if operator == 'ne':
        return value != wanted
    if operator == 'in':
        return value in wanted
    if operator == 'not_in':
        return value not in wanted
    # The four that need to know what comes before what. `index` refuses a
    # categorical aspect, which is the refusal this operator deserves.
    here = aspect.index(value)
    there = aspect.index(wanted)
    if operator == 'lt':
        return here < there
    if operator == 'lte':
        return here <= there
    if operator == 'gt':
        return here > there
    return here >= there


class LayerSelection:
    """The layers one window names, and what sits on them.

    Built by :meth:`LayerAccessor.where`. Iterating it gives the layer
    coordinates; the four properties answer the questions a window is usually
    asked, each in one pass.

    Attributes
    ----------
    layers : tuple[tuple[str, ...], ...]
        The coordinates in the window, in the graph's declaration order.
    """

    __slots__ = ('_G', 'layers')

    def __init__(self, graph, layers) -> None:
        self._G = graph
        self.layers = tuple(tuple(layer) for layer in layers)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __contains__(self, layer: object) -> bool:
        return tuple(layer) in set(self.layers) if isinstance(layer, tuple) else False

    def __repr__(self) -> str:
        return f'LayerSelection({len(self.layers)} layer(s): {list(self.layers)!r})'

    # -- what sits on the window ------------------------------------------

    @property
    def nodes(self) -> set:
        """The ids of the nodes on these layers."""
        inside = set(self.layers)
        return {vid for vid, coord in _structure.entity_keys(self._G) if coord in inside}

    @property
    def node_layers(self) -> set:
        """The ``(node_id, layer)`` keys on these layers.

        The distinction from :attr:`nodes` matters as soon as one node sits on
        two of the selected layers, which is the ordinary case.
        """
        inside = set(self.layers)
        return {key for key in _structure.entity_keys(self._G) if key[1] in inside}

    @property
    def edges(self) -> set:
        """The ids of the edges whose every endpoint is on these layers.

        Closed: an edge with one endpoint outside the window is not in it. What
        that leaves out is :attr:`crossing`.
        """
        return self._split()[0]

    @property
    def crossing(self) -> set:
        """The ids of the edges with an endpoint inside and an endpoint outside."""
        return self._split()[1]

    @property
    def boundary(self) -> set:
        """The ids of the nodes inside the window that a crossing edge touches.

        Where the window is cut. A node here has a neighbour the window does not
        hold, so an analysis over :attr:`edges` alone treats it as though it had
        none.
        """
        inside = set(self.layers)
        found = set()
        for edge_id in self.crossing:
            sides = _structure.edge_sides(self._G, edge_id)
            for item in sides.source | sides.target:
                endpoint = as_endpoint(item)
                if endpoint.layer in inside:
                    found.add(endpoint.node_id)
        return found

    def _split(self) -> tuple[set, set]:
        """One pass over the edges, giving back ``(inside, crossing)``.

        Both properties come from the same walk because asking for one and then
        the other is the common shape, and because a second pass would answer a
        question the first already had the data for.
        """
        inside_layers = set(self.layers)
        inside: set = set()
        crossing: set = set()
        for ref in _structure.iter_edges(self._G):
            sides = _structure.edge_sides(self._G, ref.id)
            members = sides.source | sides.target
            if not members:
                continue
            layers = {as_endpoint(item).layer for item in members}
            held = layers & inside_layers
            if not held:
                continue
            if held == layers:
                inside.add(ref.id)
            else:
                crossing.add(ref.id)
        return inside, crossing
