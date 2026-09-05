"""Experimental APIs for AnnNet, reachable as ``annnet.exp``.

Nothing here is part of the stable public surface. Two of these packages are
**going to leave this repository**, and the ``exp`` prefix is the point: a call
site that reads ``exp.vocabulary.check(G)`` says out loud that the import will
change.

    >>> from annnet import exp
    >>> exp.vocabulary.check(G)  # doctest: +SKIP

``vocabulary``
    What a number in a graph *means*. Reserved attribute names with declared
    domains, the identifiers two names may share, the kinds an entity may be,
    and the checks over all of it. It cannot live in the core: a general network
    data structure may not declare that ``sign`` means activation and
    inhibition, because it is not allowed to know what activation is. That rule
    is enforced by ``tests/test_core_biology_free.py``, not merely intended.

``sysbio``
    The domain workflows built on top: joining a measurement matrix onto a
    graph, projecting a result back out, and the method adapters that read one
    and write the other. Depends on ``vocabulary``.

``scverse``
    Serialising a graph *as* an AnnData, MuData or SpatialData. Generic
    interoperability rather than domain work — the opposite direction from
    ``sysbio``'s join, and it stays here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._support.lazy_exports import export_dir, resolve_lazy_export

if TYPE_CHECKING:
    from . import sysbio, scverse, vocabulary

_lazy_submodules = {
    'scverse': 'annnet.experimental.scverse',
    'sysbio': 'annnet.experimental.sysbio',
    'vocabulary': 'annnet.experimental.vocabulary',
}

__all__ = sorted(_lazy_submodules)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), name, modules=_lazy_submodules, package_name=__name__)


def __dir__() -> list[str]:
    return export_dir(globals(), __all__)
