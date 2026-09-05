"""Two names for one thing, and how a join learns they are the same.

A network names entities the way its source did. A measurement table names them
the way *its* source did. They rarely agree, and the disagreement is not noise —
it is the actual content of joining a prior knowledge network to an assay.

Three shapes of disagreement matter, and they are different problems:

**One thing, two spellings.** ``BRAF`` and ``uniprot:P15056`` denote one
molecule. Resolving that is a lookup, and the table it looks in belongs to
whoever curates it — not to this module and not to the join.

**One node, several measured things.** A complex is its subunits; a reaction is
its participants. There is no single measurement of it, so a policy has to say
what the node's value is, and there is no default worth guessing.

**One measured thing, several nodes.** A molecule appears in two compartments, or
as a transcript and a protein. Every node takes the same value; nothing is
combined.

This module holds the vocabulary for the first two. The third is the join's
business, because it is about the graph rather than about names.

Why a protocol
--------------

The join must not know that ``"A_B_C"`` is how one particular resource writes a
complex. That is a fact about that resource, and putting it in the join means the
next resource needs a second branch inside the same function. A
:class:`Mapper` is the seam: the join asks *what does this name resolve to* and
*what are this thing's members*, and every resource answers for itself.
"""

from __future__ import annotations

from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Identifier:
    """One name, in the namespace that gives it meaning.

    A bare ``"P15056"`` is ambiguous until something says which register it is
    from. Carrying the namespace makes two identifiers comparable, and makes an
    unqualified one visibly unqualified rather than silently assumed.

    Attributes
    ----------
    namespace : str | None
        Where the name is registered. ``None`` for a bare name.
    local_id : str
        The name itself.

    Examples
    --------
    >>> Identifier.parse('uniprot:P15056')
    Identifier(namespace='uniprot', local_id='P15056')
    >>> str(Identifier.parse('BRAF'))
    'BRAF'
    """

    namespace: str | None
    local_id: str

    @classmethod
    def parse(cls, text: str) -> Identifier:
        """Read ``"namespace:local_id"``, or a bare name.

        Splits on the first colon only, because a local id may contain one.
        """
        namespace, sep, local = str(text).partition(':')
        if not sep:
            return cls(None, namespace)
        return cls(namespace, local)

    def __str__(self) -> str:
        return self.local_id if self.namespace is None else f'{self.namespace}:{self.local_id}'

    @property
    def qualified(self) -> bool:
        """Whether this identifier says which register it comes from."""
        return self.namespace is not None


def parse(text: str) -> Identifier:
    """Read one identifier from ``"namespace:local_id"`` or a bare name."""
    return Identifier.parse(text)


def render(identifier: Identifier) -> str:
    """Write one identifier back as text."""
    return str(identifier)


class Mapper(Protocol):
    """What a join asks about a name.

    Two questions, and they are different. :meth:`resolve` is *what does this
    name denote*; :meth:`members` is *what is this thing made of*. A resource
    answers both for itself, so the join holds no resource-specific rule.
    """

    def resolve(self, name: str) -> list[Identifier]:
        """The identifiers one name denotes, or an empty list for none."""
        ...

    def members(self, name: str) -> list[str] | None:
        """What one composite is made of, or ``None`` when it is not composite."""
        ...


class SymbolMapper:
    """The simple mapper: a lookup table, and composites named by a separator.

    Enough for a resource that writes a complex as its subunits joined by a
    character — which is how several do — and for one that writes plain symbols,
    which is the rest.

    Parameters
    ----------
    table : Mapping[str, str | Sequence[str]], optional
        Name -> the identifier(s) it denotes. A name absent from the table
        resolves to itself, so the common case needs no table at all.
    separator : str, optional
        What joins the parts of a composite name. ``None`` means this resource
        writes no composites, and :meth:`members` always answers ``None``.
    known : Container[str], optional
        The names that exist. A composite is only split when its parts are all
        in here, so a plain name that happens to contain the separator is not
        mistaken for one — which is the failure mode splitting invites.

    Examples
    --------
    >>> mapper = SymbolMapper(separator='_', known={'A', 'B'})
    >>> mapper.members('A_B')
    ['A', 'B']
    >>> mapper.members('SLC2A1') is None
    True
    """

    __slots__ = ('_table', '_separator', '_known')

    def __init__(
        self,
        table: Any = None,
        *,
        separator: str | None = None,
        known: Any = None,
    ) -> None:
        self._table = dict(table or {})
        self._separator = separator
        self._known = known

    def resolve(self, name: str) -> list[Identifier]:
        """The identifiers one name denotes.

        A name the table does not hold resolves to itself: a mapper with no table
        is the identity, which is what a resource that already agrees needs.
        """
        found = self._table.get(name, name)
        if isinstance(found, str):
            found = [found]
        return [Identifier.parse(item) for item in found]

    def members(self, name: str) -> list[str] | None:
        """The parts of a composite name, or ``None``.

        Splits only when every part is known. A name containing the separator
        that is not a composite — and there are always some — stays whole.
        """
        if self._separator is None or self._separator not in str(name):
            return None
        parts = str(name).split(self._separator)
        if self._known is not None and not all(part in self._known for part in parts):
            return None
        return parts


class NullMapper:
    """The mapper that maps nothing: names are already identifiers.

    The default, so the simple case needs no argument and the interesting case is
    the one that has to say so.
    """

    __slots__ = ()

    def resolve(self, name: str) -> list[Identifier]:
        return [Identifier.parse(name)]

    def members(self, name: str) -> list[str] | None:
        return None


def as_mapper(value: Any) -> Mapper:
    """Return a mapper, from one or from ``None``.

    Parameters
    ----------
    value : Mapper | Mapping | None

    Returns
    -------
    Mapper
    """
    if value is None:
        return NullMapper()
    if hasattr(value, 'resolve') and hasattr(value, 'members'):
        return value
    return SymbolMapper(value)


def names_of(mapper: Mapper, name: str) -> Sequence[str]:
    """The measured names one graph-side name stands for.

    A composite stands for its members; anything else stands for itself. This is
    the one question a join needs, and it is here rather than in the join so that
    what counts as a composite stays a property of the resource.
    """
    found = mapper.members(name)
    return list(found) if found else [str(name)]
