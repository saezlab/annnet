"""Write the machine-readable public-surface inventory.

``docs/reference/public-surface.json`` lists every name the package exports,
every name a graph and a view answer to, and every public name of each
namespace. ``tests/test_public_surface.py`` checks it against the package both
ways, and ``docs/reference/api-contract.md`` is its prose. Regenerate it after
a deliberate change of the surface, never to make a test pass::

    python tools/public_surface.py            # writes the file
    python tools/public_surface.py --check    # exits 1 when the file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'reference' / 'public-surface.json'

# The namespaces the package owns. ``nx``, ``ig`` and ``gt`` are proxies onto
# other libraries, whose surface is theirs and changes with their version.
NAMESPACES = (
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
)


def _public(obj) -> list[str]:
    return sorted(name for name in dir(obj) if not name.startswith('_'))


def inventory() -> dict:
    sys.path.insert(0, str(ROOT))
    import annnet
    import annnet.core as core
    from annnet import AnnNet

    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e0')
    view = G.view()
    return {
        'annnet': sorted(annnet.__all__),
        'core': sorted(core.__all__),
        'graph': sorted(dir(G)),
        'namespaces': {name: _public(getattr(G, name)) for name in NAMESPACES},
        'view': sorted(dir(view)),
        'view_namespaces': {
            name: _public(getattr(view, name)) for name in ('attrs', 'layers', 'slices', 'N', 'E')
        },
        'selection': {
            'NodeSequence': _public(G.N),
            'EdgeSequence': _public(G.E),
            'RowSelection': _public(G.attrs.select('nodes')),
        },
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = json.dumps(inventory(), indent=2) + '\n'
    if '--check' in argv:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f'{OUT} is stale; run python tools/public_surface.py', file=sys.stderr)
            return 1
        print(f'{OUT} is current')
        return 0
    OUT.write_text(text)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
