"""The mapping between a measurement table's rows and a graph's layers.

Written once because :func:`attach`, :func:`write_back` and every method adapter
need it, and two implementations that disagree land the right numbers on the
wrong rows — which is the failure mode that looks like a result.
"""

from __future__ import annotations

from typing import Any


def column(table, key, fallback, side):
    """One column of a frame, or its index when no column is named.

    Parameters
    ----------
    table : DataFrame-like
        ``obs`` or ``var``.
    key : str | None
        The column holding the name. ``None`` means use the index.
    fallback : Iterable
        The index, used when ``key`` is ``None``.
    side : str
        ``"obs"`` or ``"var"``, for the error message.

    Returns
    -------
    list[str]

    Raises
    ------
    KeyError
        If ``key`` is not a column.
    """
    if key is None:
        return [str(value) for value in fallback]
    if key not in table.columns:
        raise KeyError(f'{side} has no column {key!r}; it has {list(table.columns)!r}')
    return [str(value) for value in table[key]]


def coordinate(aspects, aspect, value, within):
    """The layer coordinate one condition names.

    A graph with one aspect needs nothing else. A graph with more needs every
    other aspect fixed, because a coordinate is a full tuple and half of one
    addresses nothing.

    Raises
    ------
    ValueError
        If ``within`` does not fix every other aspect.
    """
    if len(aspects) == 1:
        return (str(value),)
    within = within or {}
    absent = [name for name in aspects if name != aspect and name not in within]
    if absent:
        raise ValueError(
            f'this graph declares {list(aspects)!r}, so a layer needs a value for '
            f'{sorted(absent)!r} as well. Pass within={{...}} to fix them.'
        )
    return tuple(str(value) if name == aspect else str(within[name]) for name in aspects)


def obs_layers(
    graph,
    adata: Any,
    *,
    aspect: str | None,
    layer_key: str | None = None,
    within: dict | None = None,
    called: str = 'this',
) -> list[tuple]:
    """The layer coordinate each ``obs`` row stands for, in ``obs`` order.

    Parameters
    ----------
    graph : AnnNet
    adata : anndata.AnnData
    aspect : str
        Which aspect the ``obs`` rows are values of.
    layer_key : str, optional
        The ``obs`` column holding the condition name. Default: ``obs_names``.
    within : dict, optional
        Fix the other aspects.
    called : str
        What to call the caller in an error message.

    Returns
    -------
    list[tuple]

    Raises
    ------
    ValueError
        If ``aspect`` is not given, or ``within`` does not fix every other one.
    KeyError
        If ``aspect`` is not declared, or the named column is missing.
    """
    if aspect is None:
        raise ValueError(f'{called} needs the aspect its obs rows are values of')
    declared = tuple(graph.aspects or ())
    if aspect not in declared:
        raise KeyError(f'unknown aspect {aspect!r}; this graph declares {list(declared)!r}')
    conditions = column(adata.obs, layer_key, adata.obs_names, 'obs')
    return [coordinate(declared, aspect, value, within) for value in conditions]
