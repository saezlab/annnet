"""What a graph was built from, recorded as data rather than as a memory.

A graph assembled from four downloaded tables knows everything about its own
structure and nothing about where that structure came from. The history log does
not close the gap: it records the *edits*, so reloading a file tells you what was
done and not what it was done to. A release that changed under you leaves no
trace, and neither does the difference between two runs a month apart.

So each reader records what it read: a name, a version if the source states one,
where it came from, when, and a checksum of the bytes. It goes in ``uns``, which
is graph-level state and survives every round trip the graph survives.

The record is deliberately thin. It says what was read, not what it meant — what
a source's contents mean belongs to the reader and to whatever declares the
vocabulary.
"""

from __future__ import annotations

import os
from typing import Any
from hashlib import blake2b
from datetime import UTC, datetime

#: Where the records live in ``uns``. One key, so a caller writing ``uns`` by
#: hand can see what is reserved.
KEY = '__provenance__'

#: Files above this are not hashed: the cost stops being negligible next to the
#: parse, and a missing checksum is honest where a slow one is not.
CHECKSUM_LIMIT = 512 * 1024 * 1024

#: The fields a record leads with, in the order a reader wants them.
FIELDS = ('name', 'format', 'uri', 'version', 'retrieved', 'checksum', 'reader')


def checksum_of(path, *, limit: int = CHECKSUM_LIMIT) -> str | None:
    """A content hash of one file, or ``None`` when there is nothing to hash.

    ``None`` also comes back for a file past ``limit``, which is recorded as a
    checksum that was not taken rather than one that failed.

    Parameters
    ----------
    path : str | os.PathLike
    limit : int, default :data:`CHECKSUM_LIMIT`

    Returns
    -------
    str | None
    """
    try:
        size = os.path.getsize(path)
    except (OSError, TypeError, ValueError):
        return None
    if size > limit:
        return None
    digest = blake2b(digest_size=16)
    try:
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def record(graph, name: str, *, uri=None, checksum: Any = True, **fields: Any) -> dict:
    """Record one source this graph was built from.

    Parameters
    ----------
    graph : AnnNet
    name : str
        What the source is called — the resource name, not the file name, where
        the two differ.
    uri : str | os.PathLike, optional
        Where it came from. A readable local path is hashed unless ``checksum``
        says otherwise.
    checksum : str | bool, default True
        ``True`` hashes ``uri``; ``False`` records none; a string is used as
        given, for a source that states its own.
    **fields
        ``format``, ``version``, ``reader``, and anything else worth keeping.

    Returns
    -------
    dict
        The record as stored.
    """
    entry: dict[str, Any] = {'name': str(name)}
    if uri is not None:
        entry['uri'] = str(uri)
    entry['retrieved'] = fields.pop('retrieved', None) or datetime.now(UTC).isoformat()
    if checksum is True:
        found = checksum_of(uri) if uri is not None else None
        if found is not None:
            entry['checksum'] = found
    elif checksum:
        entry['checksum'] = str(checksum)
    entry.update({key: value for key, value in fields.items() if value is not None})

    held = graph.uns.setdefault(KEY, [])
    if not isinstance(held, list):
        held = [held]
        graph.uns[KEY] = held
    held.append(entry)
    return entry


def records(graph) -> list[dict]:
    """Every source record this graph carries, in the order they were read."""
    held = graph.uns.get(KEY) or []
    if isinstance(held, dict):
        return [dict(held)]
    return [dict(entry) for entry in held]


def as_frame(graph, *, backend: str | None = None):
    """Every source record as a table."""
    from .._support.dataframe_backend import empty_dataframe, dataframe_from_rows

    found = records(graph)
    backend = backend or graph._annotations_backend
    if not found:
        return empty_dataframe(dict.fromkeys(FIELDS, 'text'), backend=backend)
    columns = list(FIELDS) + sorted({key for entry in found for key in entry} - set(FIELDS))
    rows = [{column: entry.get(column) for column in columns} for entry in found]
    return dataframe_from_rows(rows, backend=backend)


class ProvenanceAccessor:
    """What one graph was built from (``G.provenance``).

    Callable, so ``G.provenance()`` is the table and ``G.provenance.record(...)``
    is how a reader adds to it. A reader reaching for the module directly would
    be reading a private part of the core from outside it, which is what this
    accessor exists to avoid.
    """

    __slots__ = ('_G',)

    def __init__(self, graph) -> None:
        self._G = graph

    def __call__(self, *, backend: str | None = None):
        """Every source record, as a table."""
        return as_frame(self._G, backend=backend)

    def record(self, name: str, *, uri=None, checksum: Any = True, **fields: Any) -> dict:
        """Record one source this graph was built from.

        Parameters
        ----------
        name : str
            The resource name, not the file name, where the two differ.
        uri : str | os.PathLike, optional
            Where it came from. A readable local path is hashed.
        checksum : str | bool, default True
            ``True`` hashes ``uri``; ``False`` records none; a string is used as
            given.
        **fields
            ``format``, ``version``, ``reader``, and anything else worth keeping.

        Returns
        -------
        dict
            The record as stored.
        """
        return record(self._G, name, uri=uri, checksum=checksum, **fields)

    def records(self) -> list[dict]:
        """Every source record, as dicts, in the order they were read."""
        return records(self._G)

    def __len__(self) -> int:
        return len(records(self._G))

    def __iter__(self):
        return iter(records(self._G))

    def __repr__(self) -> str:
        found = records(self._G)
        return f'ProvenanceAccessor({len(found)} source(s): {[e["name"] for e in found]!r})'
