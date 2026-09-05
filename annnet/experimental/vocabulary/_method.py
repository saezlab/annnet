"""What a method needs a graph to be, written down as an object.

:func:`~annnet.experimental.vocabulary.requires` takes capability names as loose
strings. An adapter's requirement is a property of the method — the same on every
call, and readable without running anything — so it becomes a value instead::

    SPEC = MethodSpec(name='decoupler', requires_edge=('sign',), directed=True, hyperedges='none')
    exp.vocabulary.check(G, method=SPEC, strict=True)

A dataclass and not a registry: designing a lookup is for when there are enough
specs that looking one up by name is the awkward part.
"""

from __future__ import annotations

from dataclasses import field, dataclass

#: What ``hyperedges`` may say.
HYPEREDGE_POLICIES = ('allow', 'none')


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """The shape of graph one method can run on.

    Parameters
    ----------
    name : str
        What to call the method in a refusal.
    requires_edge : tuple[str, ...]
        Edge attributes every edge must carry. Presence, not domain — whether
        ``sign`` holds a legal value is the contract's question.
    requires_node : tuple[str, ...]
        Node attributes every node must carry.
    directed : bool | None
        ``True`` demands every edge run one way. ``None`` does not care.
    acyclic : bool | None
        ``True`` demands no directed cycle. ``None`` does not care.
    hyperedges : {"allow", "none"}
        ``"none"`` demands every edge join exactly two entities. A method handed
        a hyperedge it cannot read will read its projection, which is a
        different network.
    bipartite : bool
        ``True`` demands that no entity is both a source and a target.

    Notes
    -----
    Every field defaults to "does not care", and that default is load-bearing.
    **An over-declared spec is worse than none**: it refuses graphs the method
    would have handled, and what a user learns from that is to skip the check.

    Examples
    --------
    >>> SPEC = MethodSpec(name='decoupler', requires_edge=('sign',), directed=True)
    >>> SPEC.capabilities()
    ('directed',)
    """

    name: str = 'method'
    requires_edge: tuple[str, ...] = field(default=())
    requires_node: tuple[str, ...] = field(default=())
    directed: bool | None = None
    acyclic: bool | None = None
    hyperedges: str = 'allow'
    bipartite: bool = False

    def __post_init__(self) -> None:
        if self.hyperedges not in HYPEREDGE_POLICIES:
            raise ValueError(
                f'hyperedges must be one of {HYPEREDGE_POLICIES}, got {self.hyperedges!r}'
            )
        # Tuples so a spec is hashable and cannot be edited through a shared list.
        object.__setattr__(self, 'requires_edge', tuple(self.requires_edge))
        object.__setattr__(self, 'requires_node', tuple(self.requires_node))

    def capabilities(self) -> tuple[str, ...]:
        """The capability names this spec amounts to.

        The attribute requirements are not capabilities; they are checked apart.
        """
        names = []
        if self.directed:
            names.append('directed')
        if self.hyperedges == 'none':
            names.append('dyadic')
        if self.bipartite:
            names.append('bipartite')
        if self.acyclic:
            names.append('acyclic')
        return tuple(names)

    def __str__(self) -> str:
        parts = [f'{name} on every edge' for name in self.requires_edge]
        parts += [f'{name} on every node' for name in self.requires_node]
        parts += list(self.capabilities())
        return f'{self.name} needs: ' + (', '.join(parts) if parts else 'nothing in particular')
