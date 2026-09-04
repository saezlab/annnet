"""What a bulk write does about an id the graph already holds.

A batch that half-lands is worse than one that does not land: the caller sees an
exception and a graph that is neither what it was nor what it asked for. So the
policy is decided before anything is written, and ``error`` — the default —
leaves the graph untouched.

The four policies are the four honest answers. ``error`` refuses and names what
it refused. ``skip`` keeps what the graph has. ``replace`` keeps what the batch
brought. ``rename`` keeps both, giving the newcomer a suffixed id. Anything else
is one of these with the decision hidden.
"""

from __future__ import annotations

from typing import Any

#: What a bulk write may do about an id already taken.
ON_CONFLICT = ('error', 'skip', 'rename', 'replace')


class EdgeIdConflict(ValueError):
    """Raised when a bulk write meets an id the graph already holds.

    Attributes
    ----------
    ids : list[str]
        The conflicting ids, in the order the batch named them.
    """

    def __init__(self, ids, *, kind: str = 'edge') -> None:
        self.ids = list(ids)
        shown = self.ids[:5]
        more = '' if len(self.ids) <= 5 else f' (and {len(self.ids) - 5} more)'
        super().__init__(
            f'{len(self.ids)} {kind} id(s) are already taken: {shown!r}{more}. Nothing was '
            f"written. Pass on_conflict='skip' to keep what the graph has, 'replace' to "
            f"keep what this batch brings, or 'rename' to keep both."
        )


def _free_id(taken, wanted: str, prefix: str) -> str:
    """A suffixed id nothing holds, derived from the one that was taken."""
    index = 2
    candidate = f'{wanted}~{index}'
    while candidate in taken:
        index += 1
        candidate = f'{wanted}~{index}'
    return candidate


def resolve_conflicts(
    graph,
    specs: list[dict],
    ids: list[str],
    *,
    on_conflict: str = 'error',
    prefix: str = 'e',
    kind: str = 'edge',
) -> list[dict]:
    """Apply a conflict policy to a batch, and return the specs still to write.

    ``ids`` is edited in place where a policy changes an id, so the caller's
    row-order mapping stays true whatever the policy did.

    Parameters
    ----------
    graph : AnnNet
    specs : list[dict]
        The batch, each carrying an ``edge_id``.
    ids : list[str]
        The same ids, in row order. Rewritten in place under ``"rename"``.
    on_conflict : str, default "error"
        One of :data:`ON_CONFLICT`.
    prefix : str, default "e"
        Used when a renamed id needs one.
    kind : str, default "edge"
        What to call the element in a refusal.

    Returns
    -------
    list[dict]
        The specs to write. Empty when every one of them was skipped.

    Raises
    ------
    EdgeIdConflict
        Under ``"error"``, when any id is taken. Nothing is written.
    """
    held = {str(existing) for existing in graph.edges()}
    clashing = [spec['edge_id'] for spec in specs if spec['edge_id'] in held]
    if not clashing:
        return specs

    if on_conflict == 'error':
        raise EdgeIdConflict(clashing, kind=kind)
    if on_conflict == 'skip':
        return [spec for spec in specs if spec['edge_id'] not in held]
    if on_conflict == 'replace':
        graph.remove_edges(clashing)
        return specs

    # rename: keep both, and tell the caller what the newcomer is called by
    # rewriting the id list they were handed.
    taken = set(held)
    out: list[dict] = []
    for position, spec in enumerate(specs):
        wanted = spec['edge_id']
        if wanted in taken:
            renamed = _free_id(taken, wanted, prefix)
            spec = {**spec, 'edge_id': renamed}
            ids[position] = renamed
            wanted = renamed
        taken.add(wanted)
        out.append(spec)
    return out


def require_policy(value: Any) -> str:
    """Return ``value`` if it names a conflict policy, and raise otherwise."""
    if value not in ON_CONFLICT:
        raise ValueError(f'on_conflict must be one of {ON_CONFLICT}, got {value!r}')
    return value
