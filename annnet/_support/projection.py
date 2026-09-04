"""How a hyperedge becomes binary edges, named once for every exporter.

Most formats hold pairs. A hyperedge is not a pair, so every exporter has to
decide what to do with one, and each of them grew its own spelling: the adapters
say ``hyperedge_mode``, CX2 says ``hyperedges``, and the words behind them were
documented three times and agreed by coincidence.

This is the one place that vocabulary lives. The words are the ones already in
use — renaming them would trade two spellings for three — and the two names the
literature prefers are accepted as aliases, because *star* and *clique* say what
``reify`` and ``expand`` do.

What a projection costs
-----------------------

``skip`` loses the hyperedge. ``expand`` loses *which* members were one relation
— a four-member hyperedge becomes six pairs that no longer say they belong
together. ``reify`` keeps that by adding a node standing for the relation, at the
cost of a node that is not an entity.

None of the three is lossless, which is why a caller picks rather than a default
deciding. What ``expand`` also silently loses is a **per-member coefficient**:
those live on the hyperedge, and a pair between two members has nowhere to put
one. That loss is the only one loud enough to refuse by default.
"""

from __future__ import annotations

from typing import Any

#: How a hyperedge may be projected onto binary edges.
#:
#: ``skip`` drops it. ``reify`` adds a node standing for the relation and joins
#: every member to it — the *star*. ``expand`` joins every pair of members —
#: the *clique*.
PROJECTIONS = ('skip', 'reify', 'expand')

#: The names the multilayer literature uses for the same two shapes, accepted so
#: a caller may write either. One vocabulary, two spellings — not three words.
ALIASES = {'star': 'reify', 'clique': 'expand'}

#: What a caller may say about per-member coefficients a projection cannot carry.
COEFFICIENT_POLICIES = ('error', 'drop')


class CoefficientsWouldBeLost(ValueError):
    """Raised when a projection would silently drop per-member coefficients.

    A directed hyperedge may weight each member separately. ``expand`` turns it
    into pairs, and a pair has nowhere to record what one member's coefficient
    was — so the numbers vanish and the file looks complete.

    Attributes
    ----------
    edge_ids : list[str]
        The hyperedges carrying coefficients the projection cannot hold.
    """

    def __init__(self, edge_ids, projection: str) -> None:
        self.edge_ids = list(edge_ids)
        shown = self.edge_ids[:5]
        more = '' if len(self.edge_ids) <= 5 else f' (and {len(self.edge_ids) - 5} more)'
        super().__init__(
            f'{len(self.edge_ids)} hyperedge(s) carry per-member coefficients that a '
            f'{projection!r} projection cannot hold: {shown!r}{more}. Pass '
            f"coefficients='drop' to accept the loss, or hyperedges='reify' to keep "
            f'the relation as a node that can carry them.'
        )


def normalise(value: Any, *, default: str = 'skip') -> str:
    """Return one projection name, resolving an alias.

    Parameters
    ----------
    value : str | None
        A name from :data:`PROJECTIONS` or :data:`ALIASES`. ``None`` gives
        ``default``.
    default : str, default "skip"
        What ``None`` means.

    Returns
    -------
    str
        One of :data:`PROJECTIONS`.

    Raises
    ------
    ValueError
        If the name is neither a projection nor an alias.

    Examples
    --------
    >>> normalise('clique')
    'expand'
    >>> normalise(None, default='reify')
    'reify'
    """
    if value is None:
        return default
    found = ALIASES.get(value, value)
    if found not in PROJECTIONS:
        raise ValueError(
            f'hyperedges must be one of {PROJECTIONS} (or {tuple(ALIASES)}), got {value!r}'
        )
    return found


def check_coefficients(edge_ids, projection: str, coefficients: str = 'error') -> None:
    """Refuse a projection that would drop per-member coefficients.

    Takes the ids rather than the graph: this module holds the vocabulary and
    nothing that reads a graph, so that ``_support`` stays a leaf.
    :func:`annnet.core._structure.hyperedges_with_coefficients` is what names
    them.

    Only ``expand`` loses them — ``skip`` keeps nothing and says so, and
    ``reify`` keeps the relation as a node that can carry them.

    Parameters
    ----------
    edge_ids : Sequence[str]
        The hyperedges that weight their members separately.
    projection : str
        A normalised name from :data:`PROJECTIONS`.
    coefficients : {"error", "drop"}, default "error"
        ``"drop"`` accepts the loss.

    Raises
    ------
    ValueError
        If ``coefficients`` is not one of :data:`COEFFICIENT_POLICIES`.
    CoefficientsWouldBeLost
        Under ``"error"``, when any id was given.
    """
    if coefficients not in COEFFICIENT_POLICIES:
        raise ValueError(
            f'coefficients must be one of {COEFFICIENT_POLICIES}, got {coefficients!r}'
        )
    if projection != 'expand' or coefficients == 'drop':
        return
    if edge_ids:
        raise CoefficientsWouldBeLost(edge_ids, projection)
