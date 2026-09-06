"""Adapters onto established method packages.

Each one declares what it needs before it runs, reads its input off the graph,
and writes its result back **additively** — a new slice plus new attributes on
the object that was already there, so a prior and two different fits coexist and
can be diffed.

**No adapter reimplements any arithmetic.** That belongs to the method package,
and each adapter has a test pinning numeric parity against calling it directly on
the equivalent DataFrame. What an adapter contributes is the declaration before,
the reading off the graph, and the write-back after.

Each needs its method package installed::

    pip install 'annnet[decoupler]'   # or annnet[corneto], or annnet[methods]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...._support.lazy_exports import export_dir, resolve_lazy_export

if TYPE_CHECKING:
    from . import corneto, decoupler

_lazy_submodules = {
    'corneto': 'annnet.experimental.sysbio.methods.corneto',
    'decoupler': 'annnet.experimental.sysbio.methods.decoupler',
}

__all__ = sorted(_lazy_submodules)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), name, modules=_lazy_submodules, package_name=__name__)


def __dir__() -> list[str]:
    return export_dir(globals(), __all__)
