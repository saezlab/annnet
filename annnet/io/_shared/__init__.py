"""What every format adapter in :mod:`annnet.io` shares.

The package splits in two. These modules are the plumbing — the destination
contract a reader ends in, the sidecar a lossy writer leaves behind, the archive
helpers, and the vocabulary translation between stored words and the core's. The
modules beside this package are the formats themselves, one per family, and each
one knows about its own file layout and nothing else.

A format module imports from here. Nothing here imports a format.
"""

from __future__ import annotations
