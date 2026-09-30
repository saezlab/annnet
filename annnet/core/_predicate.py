"""The one predicate engine every selection surface shares.

``G.N.select(score__gt=0.5)``, ``G.E.select(kind='hyper')``,
``G.attrs.select('node_layers', score__gte=0.8)`` and
``G.layers.where(time__lte='12h')`` are four spellings of one question: which
rows of this axis satisfy these conditions. They parse the conditions here,
evaluate them here, and report a bad field, operator or value the same way.

A condition is ``field=value`` (equality) or ``field__operator=value`` with one
of :data:`OPERATORS`. A literal field whose own name contains ``__`` wins over
suffix parsing, and a positional mapping may use explicit ``(field, operator)``
keys where a keyword would be ambiguous.

Null handling is one rule everywhere: a missing value — ``None`` or a float
``NaN`` — is excluded by every comparison, including ``ne`` and ``not_in``, and
selected only by an explicit ``field__isnull=True``.

An ordered aspect compares by its declared order, through ``order=``; a
categorical aspect refuses the four ordering operators. A numeric attribute
compares numerically. Two values that cannot be compared raise, naming the
field and the operator, rather than silently dropping the row.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from collections.abc import Mapping, Iterable

import numpy as np

#: The comparison operators. ``eq`` is what a bare ``field=value`` means.
OPERATORS = ('eq', 'ne', 'in', 'not_in', 'lt', 'lte', 'gt', 'gte', 'isnull')

#: The operators that ask where a value sits rather than what it is. Only an
#: ordered aspect, or a value type with an order of its own, can answer these.
ORDERED_ONLY = ('lt', 'lte', 'gt', 'gte')

_MEMBERSHIP = ('in', 'not_in')


class Condition(NamedTuple):
    """One parsed condition: the field, the operator and the wanted value."""

    field: str
    operator: str
    value: Any


def is_missing(value) -> bool:
    """Return True for the two spellings of "no value": ``None`` and a NaN."""
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    return False


def _collection(value, *, field: str, operator: str) -> frozenset | tuple:
    """Return the values a membership operator compares against, hashed when possible."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            f"field {field!r}: '{operator}' needs a collection of values, got {type(value).__name__}"
        )
    items = tuple(value)
    try:
        return frozenset(items)
    except TypeError:
        return items


def parse_conditions(
    conditions: Mapping | None = None,
    /,
    *,
    fields: Iterable[str] | None = None,
    noun: str = 'field',
    **keywords,
) -> list[Condition]:
    """Parse conditions into :class:`Condition` triples.

    Parameters
    ----------
    conditions : Mapping, optional
        A positional mapping. A key is a keyword as below, or an explicit
        ``(field, operator)`` tuple.
    fields : Iterable[str], optional
        The fields the axis knows. A field that is not one of them raises,
        naming the known fields. A field whose name holds ``__`` is matched
        literally before any suffix is read off it. ``None`` skips the check.
    noun : str, default "field"
        What a field is called in the error message (``"aspect"`` for a layer
        window).
    **keywords
        ``field=value`` or ``field__operator=value``.

    Returns
    -------
    list[Condition]

    Raises
    ------
    KeyError
        If a field is not one of ``fields``.
    ValueError
        If an operator is not one of :data:`OPERATORS`.
    TypeError
        If the value does not fit the operator (a membership operator needs a
        collection, ``isnull`` needs a bool).
    """
    known = None if fields is None else set(fields)
    items: list = []
    if conditions:
        items.extend(conditions.items())
    items.extend(keywords.items())

    parsed: list[Condition] = []
    for key, value in items:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError(
                    f'a condition key is a name or a (field, operator) pair, not {key!r}'
                )
            field, operator = key
            if operator not in OPERATORS:
                raise ValueError(
                    f'unknown operator {operator!r} for field {field!r}; one of {list(OPERATORS)} is expected'
                )
            if known is not None and field not in known:
                raise KeyError(_unknown_field(field, known, noun))
        elif known is not None and key in known:
            field, operator = key, 'eq'
        else:
            name, _, suffix = key.rpartition('__')
            if name and suffix in OPERATORS:
                field, operator = name, suffix
            elif name and known is None:
                # No schema to check against and an unknown suffix: the whole
                # key is the field.
                field, operator = key, 'eq'
            elif name and known is not None and name in known:
                raise ValueError(
                    f'unknown operator {suffix!r} in {key!r}; one of {list(OPERATORS)} is expected'
                )
            else:
                field, operator = key, 'eq'
            if known is not None and field not in known:
                raise KeyError(_unknown_field(field, known, noun))
        parsed.append(Condition(field, operator, _checked_value(field, operator, value)))
    return parsed


def _unknown_field(field, known, noun: str = 'field') -> str:
    if noun == 'aspect':
        return f'unknown aspect {field!r}; this graph declares {sorted(known, key=str)!r}'
    return f'unknown {noun} {field!r}; this axis knows {sorted(known, key=str)!r}'


def _checked_value(field: str, operator: str, value):
    if operator in _MEMBERSHIP:
        return _collection(value, field=field, operator=operator)
    if operator == 'isnull':
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f'field {field!r}: isnull takes True or False, got {value!r}')
        return bool(value)
    return value


def _compare_error(field, operator, value, wanted, cause):
    return TypeError(
        f'field {field!r}: cannot compare {value!r} ({type(value).__name__}) with '
        f'{wanted!r} ({type(wanted).__name__}) using {operator!r}'
    )


def satisfies(operator: str, value, wanted, *, field: str = '?', order=None) -> bool:
    """Whether one value satisfies one condition.

    Parameters
    ----------
    operator : str
        One of :data:`OPERATORS`.
    value : Any
        What the row holds for the field. ``None`` or NaN means nothing.
    wanted : Any
        What the condition asked for. Already checked by
        :func:`parse_conditions` for the membership and null operators.
    field : str, optional
        For the error message.
    order : Aspect, optional
        The declared order of the values, for an ordered aspect. When given,
        the four ordering operators compare positions in that order rather
        than the values themselves.

    Raises
    ------
    TypeError
        If the two values cannot be compared with this operator.
    ValueError
        If an ordering operator is asked of a categorical aspect.
    KeyError
        If ``wanted`` is not a value of the ordered aspect.
    """
    missing = is_missing(value)
    if operator == 'isnull':
        return missing is bool(wanted)
    if missing:
        return False
    if operator == 'eq':
        return _equal(value, wanted)
    if operator == 'ne':
        return not _equal(value, wanted)
    if operator == 'in':
        return _member(value, wanted)
    if operator == 'not_in':
        return not _member(value, wanted)
    # The four that need an order.
    if order is not None:
        here = order.index(value)
        there = order.index(wanted)
    else:
        here, there = value, wanted
    try:
        if operator == 'lt':
            return bool(here < there)
        if operator == 'lte':
            return bool(here <= there)
        if operator == 'gt':
            return bool(here > there)
        if operator == 'gte':
            return bool(here >= there)
    except TypeError as exc:
        raise _compare_error(field, operator, value, wanted, exc) from None
    raise ValueError(f'unknown operator {operator!r}')


def _equal(value, wanted) -> bool:
    try:
        result = value == wanted
    except Exception:  # noqa: BLE001 - an exotic __eq__ is not a match
        return False
    if isinstance(result, np.ndarray):
        return bool(result.all()) if result.size else False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _member(value, wanted) -> bool:
    try:
        return value in wanted
    except TypeError:
        return any(_equal(value, item) for item in wanted)


# ---------------------------------------------------------------------------
# Evaluating many rows at once
# ---------------------------------------------------------------------------

_NUMERIC_KINDS = 'biuf'
_VECTOR_OPS = {
    'eq': np.equal,
    'ne': np.not_equal,
    'lt': np.less,
    'lte': np.less_equal,
    'gt': np.greater,
    'gte': np.greater_equal,
}


def column_mask(condition: Condition, column, *, order=None) -> np.ndarray:
    """Return a boolean mask over one column for one condition.

    A numeric numpy column and a numeric wanted value take one vectorized
    pass, with NaN cells excluded as missing. Everything else is evaluated
    cell by cell through :func:`satisfies`, so the two paths cannot disagree
    on what a missing value means.
    """
    field, operator, wanted = condition
    values = column if isinstance(column, np.ndarray) else np.asarray(column, dtype=object)
    if (
        order is None
        and values.dtype.kind in _NUMERIC_KINDS
        and operator in _VECTOR_OPS
        and isinstance(wanted, (int, float, np.integer, np.floating, bool, np.bool_))
    ):
        with np.errstate(invalid='ignore'):
            mask = _VECTOR_OPS[operator](values, wanted)
        if values.dtype.kind == 'f':
            mask &= ~np.isnan(values)
        return np.asarray(mask, dtype=bool)
    if order is None and values.dtype.kind in _NUMERIC_KINDS and operator == 'isnull':
        return np.isnan(values) if values.dtype.kind == 'f' else np.zeros(values.shape, dtype=bool)
    if order is None and values.dtype.kind == 'f' and operator in _MEMBERSHIP:
        try:
            wanted_array = np.asarray(sorted(wanted), dtype=float)
        except (TypeError, ValueError):
            wanted_array = None
        if wanted_array is not None:
            mask = np.isin(values, wanted_array)
            if operator == 'not_in':
                mask = ~mask
            return mask & ~np.isnan(values)
    if order is None and values.dtype.kind == 'O':
        fast = _object_column_mask(operator, values, wanted)
        if fast is not None:
            return fast
    out = np.empty(values.shape[0], dtype=bool)
    for position, value in enumerate(values.tolist() if values.dtype.kind != 'O' else values):
        out[position] = satisfies(operator, value, wanted, field=field, order=order)
    return out


_PLAIN_SCALARS = (str, bytes, int, float, bool, np.integer, np.floating, np.bool_)


def _object_column_mask(operator: str, values: np.ndarray, wanted) -> np.ndarray | None:
    """One pass over an object column for a plain scalar or a set of them.

    This is :func:`satisfies` with the call removed from the inner loop: the
    same missing rule (a missing cell matches nothing but ``isnull``) and the
    same comparison. A cell whose ``==`` or ``in`` misbehaves — an array, a
    type that refuses the comparison — hands the column back to the general
    path, which judges every cell on its own.
    """
    items = values.tolist()
    count = len(items)
    if operator == 'isnull':
        missing = np.fromiter((is_missing(v) for v in items), dtype=bool, count=count)
        return missing if wanted else ~missing
    if operator in ('eq', 'ne'):
        if not isinstance(wanted, _PLAIN_SCALARS) or is_missing(wanted):
            return None
        try:
            hits = np.fromiter((v == wanted for v in items), dtype=bool, count=count)
        except (TypeError, ValueError):
            return None
        if operator == 'eq':
            return hits
        missing = np.fromiter((is_missing(v) for v in items), dtype=bool, count=count)
        return ~hits & ~missing
    if operator in _MEMBERSHIP and isinstance(wanted, frozenset):
        try:
            hits = np.fromiter((v in wanted for v in items), dtype=bool, count=count)
        except TypeError:
            return None
        if operator == 'in':
            return hits
        missing = np.fromiter((is_missing(v) for v in items), dtype=bool, count=count)
        return ~hits & ~missing
    return None


def rows_matching(
    rows: Iterable[tuple[Any, Mapping]], conditions: list[Condition], *, orders=None
) -> list:
    """Return the keys of the ``(key, row)`` pairs that satisfy every condition.

    ``orders`` maps a field to the ordered aspect it compares by, for the rows
    of an address whose fields include an aspect.
    """
    orders = orders or {}
    if not conditions:
        return [key for key, _row in rows]
    kept = []
    for key, row in rows:
        for field, operator, wanted in conditions:
            if not satisfies(
                operator, row.get(field), wanted, field=field, order=orders.get(field)
            ):
                break
        else:
            kept.append(key)
    return kept


def describe(conditions: list[Condition]) -> str:
    """One readable line for a repr: ``score>0.5, group in {'x','y'}``."""
    symbols = {
        'eq': '=',
        'ne': '!=',
        'lt': '<',
        'lte': '<=',
        'gt': '>',
        'gte': '>=',
        'in': ' in ',
        'not_in': ' not in ',
        'isnull': ' is null=',
    }
    parts = []
    for field, operator, value in conditions:
        shown = sorted(value, key=repr) if isinstance(value, frozenset) else value
        parts.append(f'{field}{symbols[operator]}{shown!r}')
    return ', '.join(parts)


__all__ = [
    'OPERATORS',
    'ORDERED_ONLY',
    'Condition',
    'column_mask',
    'describe',
    'is_missing',
    'parse_conditions',
    'rows_matching',
    'satisfies',
]
