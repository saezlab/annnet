"""An aspect, and whether its values come one before another.

A layer coordinate is one label per aspect. Some aspects are *categorical* — a
mechanism, a compartment, a name — where no value comes before another. Some are
*ordinal* — a timepoint, a dose, a stage — where they do.

The distinction is not decoration. It is what separates the two families of
coupling a multilayer graph has: an ordinal aspect couples consecutive values, a
categorical one couples across values. It is also what makes a window a query
rather than arithmetic: ``where(time__lte='12h')`` needs to know that ``'1h'``
comes before ``'12h'``, and a bare list does not say so.

Declared as a bare list, an aspect carried no order at all, so the ordering lived
in a module-level tuple beside the graph and was recomputed by hand wherever it
was needed — ``TIMES.index(t)`` for a sort, ``zip(TIMES, TIMES[1:])`` for the
pairs, ``i / (len(TIMES) - 1)`` for a feature. Three spellings of one fact, none
of them held by the object that depends on it.

A bare list still works and means ``ordered=False``, so nothing that declared
aspects the old way changes.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

#: Where an aspect's orderedness is kept, in that aspect's attributes. Reserved,
#: so a caller writing aspect attributes by hand can see what is taken.
ORDERED_KEY = '__ordered__'


class OrderedLabels:
    """A set of labels that remembers the order they were declared in.

    A ``set`` was here before, which lost that order: a graph declared with
    ``['basal', 'stim', 'late']`` read its layers back as
    ``['basal', 'late', 'stim']``, and for an ordinal aspect that is not a
    cosmetic difference — it is the wrong order, silently.

    A ``dict`` is an ordered set with the same membership cost, so this is one
    wrapping thin enough to leave every call site unchanged.
    """

    __slots__ = ('_items',)

    def __init__(self, items: Any = ()) -> None:
        self._items = dict.fromkeys(items)

    def add(self, label: str) -> None:
        """Append one label, keeping the position of one already held."""
        self._items.setdefault(label)

    def discard(self, label: str) -> None:
        """Drop one label if it is held."""
        self._items.pop(label, None)

    def update(self, labels: Any) -> None:
        """Append many labels, keeping the positions of those already held."""
        for label in labels:
            self._items.setdefault(label)

    def __contains__(self, label: object) -> bool:
        return label in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedLabels):
            return list(self) == list(other)
        if isinstance(other, (set, frozenset)):
            return set(self._items) == other
        if isinstance(other, (list, tuple)):
            return list(self._items) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f'OrderedLabels({list(self._items)!r})'


class Aspect:
    """One aspect of a multilayer graph, and whether its values are ordered.

    Parameters
    ----------
    values : Sequence[str]
        The elementary labels, in the order they are meant to be read.
    ordered : bool, default False
        Whether one value comes before another. An ordered aspect answers
        :meth:`index`, :meth:`consecutive_pairs`, :meth:`normalized_position`,
        :meth:`before` and :meth:`after`; a categorical one refuses them, because
        the answer would be the declaration order pretending to be a meaning.

    Examples
    --------
    >>> time = Aspect(['0h', '1h', '12h', '24h'], ordered=True)
    >>> time.index('12h')
    2
    >>> time.consecutive_pairs()
    [('0h', '1h'), ('1h', '12h'), ('12h', '24h')]
    >>> time.normalized_position('12h')
    0.6666666666666666
    >>> time.before('12h')
    ['0h', '1h']
    """

    __slots__ = ('values', 'ordered', '_position')

    def __init__(self, values: Sequence[str], ordered: bool = False) -> None:
        self.values = tuple(dict.fromkeys(str(value) for value in values))
        self.ordered = bool(ordered)
        self._position = {value: index for index, value in enumerate(self.values)}

    # -- membership, which either kind answers ----------------------------

    def __contains__(self, value: object) -> bool:
        return value in self._position

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Aspect):
            return self.values == other.values and self.ordered == other.ordered
        if isinstance(other, (list, tuple)):
            return list(self.values) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f'Aspect({list(self.values)!r}, ordered={self.ordered})'

    # -- order, which only an ordered aspect answers ----------------------

    def _require_order(self, what: str) -> None:
        if not self.ordered:
            raise ValueError(
                f'{what} needs an ordered aspect, and this one is categorical. '
                f'Declare it as Aspect({list(self.values)!r}, ordered=True) if its '
                f'values do come one before another.'
            )

    def _require_value(self, value: str) -> int:
        try:
            return self._position[value]
        except KeyError:
            raise KeyError(
                f'{value!r} is not a value of this aspect; it holds {list(self.values)!r}'
            ) from None

    def index(self, value: str) -> int:
        """The position of one value.

        Raises
        ------
        ValueError
            If this aspect is categorical.
        KeyError
            If the value is not one of this aspect's.
        """
        self._require_order('index')
        return self._require_value(value)

    def consecutive_pairs(self) -> list[tuple[str, str]]:
        """Every ``(value, next value)`` pair, which is what ordinal coupling couples."""
        self._require_order('consecutive_pairs')
        return list(zip(self.values, self.values[1:], strict=False))

    def normalized_position(self, value: str) -> float:
        """The position of one value scaled onto ``[0, 1]``.

        A one-value aspect answers ``0.0`` rather than dividing by zero.
        """
        self._require_order('normalized_position')
        position = self._require_value(value)
        span = len(self.values) - 1
        return position / span if span else 0.0

    def before(self, value: str, inclusive: bool = False) -> list[str]:
        """The values that come before one value."""
        self._require_order('before')
        position = self._require_value(value)
        return list(self.values[: position + 1 if inclusive else position])

    def after(self, value: str, inclusive: bool = False) -> list[str]:
        """The values that come after one value."""
        self._require_order('after')
        position = self._require_value(value)
        return list(self.values[position if inclusive else position + 1 :])


def as_aspect(value: Any) -> Aspect:
    """Return one aspect declaration as an :class:`Aspect`.

    A bare sequence is the shorthand for a categorical aspect, which is what
    every declaration written before :class:`Aspect` existed is.

    Parameters
    ----------
    value : Aspect | Sequence[str]

    Returns
    -------
    Aspect
    """
    if isinstance(value, Aspect):
        return value
    return Aspect(list(value), ordered=False)
