"""List every public name a caller can reach on a graph, a view and a namespace.

``dir()`` on an ``AnnNet`` answers with the curated ``_PUBLIC_API``, so a name
that the class carries without a leading underscore, and that the list does
not hold, is public to a caller and invisible to the inventory that ``dir()``
builds (``tools/public_surface.py``). This script walks the classes and the
instances instead of asking ``dir()``, so those names show up. Each row says
who holds the name, what kind of name it is, which class defines it, and
whether ``dir()`` lists it::

    python tools/surface_inventory.py                 # the table, on stdout
    python tools/surface_inventory.py --out FILE      # the table, into a file
    python tools/surface_inventory.py --hidden-only   # only what dir() hides

The table is the input of the contract test: a name on it is stated in
``dev/api-contract.md`` or carries a leading underscore. Nothing here measures
anything; the name of the directory is about tools, not about speed.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import functools

ROOT = Path(__file__).resolve().parents[1]

# The holders, in the order they appear in the table. ``nx``, ``ig`` and ``gt``
# are proxies onto other libraries, whose surface is theirs and changes with
# their version, so they are not walked.
GRAPH_NAMESPACES = (
    'attrs',
    'layers',
    'slices',
    'ops',
    'history',
    'provenance',
    'spaces',
    'idx',
    'cache',
    'matrices',
    'N',
    'E',
)


def _kind(holder: object, name: str) -> tuple[str, str]:
    """What kind of name this is, and the class that defines it.

    The class attribute answers both without touching the instance, which
    keeps a property from running and a removed name from raising.
    """
    for klass in type(holder).__mro__:
        if name in vars(klass):
            value = vars(klass)[name]
            if isinstance(value, (property, functools.cached_property)):
                return 'property', klass.__name__
            if isinstance(value, staticmethod):
                return 'staticmethod', klass.__name__
            if isinstance(value, classmethod):
                return 'classmethod', klass.__name__
            if callable(value):
                return 'method', klass.__name__
            if hasattr(value, '__get__'):
                return 'descriptor', klass.__name__
            return 'attribute', klass.__name__
    # Not on any class: it is in the instance dictionary.
    value = vars(holder).get(name)
    return ('method' if callable(value) else 'attribute'), '(instance)'


def _candidates(holder: object) -> set[str]:
    """Every name without a leading underscore that the holder might answer to.

    The union of the class dictionaries along the method resolution order, the
    instance dictionary, and ``dir()`` itself, because a class is free to add
    to ``dir()`` what it resolves through ``__getattr__``.
    """
    names: set[str] = set()
    for klass in type(holder).__mro__:
        if klass is object:
            continue
        names.update(vars(klass))
    try:
        names.update(vars(holder))
    except TypeError:  # an object without a __dict__
        pass
    names.update(dir(holder))
    return {name for name in names if not name.startswith('_')}


def _reachable(holder: object) -> set[str]:
    """The candidates a caller really gets a value for.

    A removed name is a descriptor that raises ``AttributeError`` with its
    replacement, so it is on the class and off the surface. Asking for the
    value is the only way to tell the two apart. Any other failure is the
    name answering in its own way, so it stays.
    """
    reachable = set()
    for name in _candidates(holder):
        try:
            getattr(holder, name)
        except AttributeError:
            continue
        except Exception:  # noqa: BLE001 - the name exists; it needs other state
            pass
        reachable.add(name)
    return reachable


def rows(graph: object, view: object) -> list[tuple[str, str, str, str, str]]:
    """One row per reachable public name: holder, name, kind, class, visibility."""
    holders: list[tuple[str, object]] = [('G', graph)]
    holders += [(f'G.{name}', getattr(graph, name)) for name in GRAPH_NAMESPACES]
    holders.append(('V', view))

    table = []
    for label, holder in holders:
        listed = set(dir(holder))
        for name in sorted(_reachable(holder)):
            kind, klass = _kind(holder, name)
            table.append(
                (label, name, kind, klass, 'listed' if name in listed else 'hidden'),
            )
    return table


def _graph():
    sys.path.insert(0, str(ROOT))
    from annnet import AnnNet

    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e0')
    return G


def render(table, *, version: str) -> str:
    """The table as text, with a header and a count per holder."""
    widths = [max(len(row[column]) for row in table) for column in range(5)]
    lines = [
        f'# The public names reachable on an AnnNet, its namespaces and a view, at annnet {version}.',
        '# Written by tools/surface_inventory.py. A hidden name is one that dir() does not list.',
        '',
    ]
    header = ('holder', 'name', 'kind', 'defined by', 'dir()')
    lines.append('  '.join(text.ljust(widths[i]) for i, text in enumerate(header)).rstrip())
    lines.append('  '.join('-' * widths[i] for i in range(5)))
    for row in table:
        lines.append('  '.join(text.ljust(widths[i]) for i, text in enumerate(row)).rstrip())

    lines += ['', '# Count per holder: total, and how many of them dir() hides.']
    for label in dict.fromkeys(row[0] for row in table):
        held = [row for row in table if row[0] == label]
        hidden = sum(1 for row in held if row[4] == 'hidden')
        lines.append(f'# {label}: {len(held)} public, {hidden} hidden')
    lines.append(
        f'# total: {len(table)} public, {sum(1 for r in table if r[4] == "hidden")} hidden'
    )
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    """Write the table, or print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, help='write the table here instead of stdout')
    parser.add_argument(
        '--hidden-only',
        action='store_true',
        help='keep only the names that dir() does not list',
    )
    args = parser.parse_args(argv)

    import annnet

    graph = _graph()
    table = rows(graph, graph.view())
    if args.hidden_only:
        table = [row for row in table if row[4] == 'hidden']
    text = render(table, version=annnet.__version__)
    if args.out:
        args.out.write_text(text)
        print(f'wrote {args.out}')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
